from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# --- event tags used by the optional debug callback ----------------------
EVT_COMM_UPDATE   = "comm_update"
EVT_GAMMA1_CROSS  = "gamma1_cross"
EVT_GAMMA2_DIRECT = "gamma2_direct"
EVT_CONF_UPDATE   = "conf_update"
EVT_ACK           = "ack"
EVT_NACK          = "nack"
EVT_DONE          = "done"


@dataclass
class BAMConfig:
    gamma_1: float = 0.85
    gamma_2: float = 0.99
    rho_ack: float = 0.95
    rho_nack: float = 0.95
    p_field: int = 4
    # Mixture-likelihood: prior probability that an observed token is
    # uninformative (the watermark could not bias it, e.g. a low-entropy
    # step). Matches EPS_NOISE in compare_chat.py. In [0, 1).
    eps_noise: float = 0.5


@dataclass
class BAMTracker:
    M: int
    cfg: BAMConfig
    channel_angle: Callable[[int, int], float]
    coupled_dist: Callable[[np.ndarray, int, int], np.ndarray]
    angle_of_token: Callable[[int, int], float]
    phi: float = 0.0   # the global ArcMark phase
    sample_from: Callable[[np.ndarray], int] = None
    t_offset: int = 0
    on_event: Optional[Callable[[str, dict], None]] = None

    # state
    pi: np.ndarray = field(init=False)
    knockdown: np.ndarray = field(init=False)
    phase: str = field(default="COMM", init=False)
    candidate: Optional[int] = field(default=None, init=False)
    rho: Optional[np.ndarray] = field(default=None, init=False)
    rng: np.random.Generator = field(init=False)
    done: bool = field(default=False, init=False)
    decoded: Optional[int] = field(default=None, init=False)
    t: int = field(default=0, init=False)              # steps consumed
    n_comm: int = field(default=0, init=False)
    n_conf: int = field(default=0, init=False)

    # mixture-likelihood constants (derived from cfg in __post_init__)
    _sigma: float = field(init=False)
    _z_signal: float = field(init=False)

    def __post_init__(self):
        self.pi = np.ones(self.M) / self.M
        self.knockdown = np.ones(self.M)
        self.rng = np.random.default_rng()
        # signal width = angular noise of an informative token; normaliser
        # for the (non-wrapped) Gaussian, matching compare_chat.py exactly.
        self._sigma = math.pi / self.cfg.p_field
        self._z_signal = math.sqrt(2.0 * math.pi) * self._sigma

    # ---- internal: belief update from one observation ------------------
    def _per_symbol_likelihood(self, angle_obs: float) -> np.ndarray:
        """Robust per-symbol likelihood from the observed angle ALONE.

        Mixture model (see compare_chat.py): a normalised wrapped-Gaussian
        signal term plus a uniform floor.

            ell_u = (1 - eps) * exp(-d(angle, mu_u)^2 / 2 sigma^2) / Z
                    + eps / (2 pi)

        The uniform floor eps/(2 pi) models "with probability eps this token
        was uninformative". When the observed angle sits BETWEEN all symbol
        targets (a noise token), the signal term is tiny for every u, so the
        likelihood is ~flat across symbols and the multiplicative posterior
        update leaves the belief almost unchanged instead of injecting
        confident noise. When the angle sits NEAR a target, the signal term
        dominates and the token is informative.

        Decoder-computable: depends only on angle_obs (token + shared secret)
        and public constants. No distribution / model access.
        """
        p = self.cfg.p_field
        eps = self.cfg.eps_noise
        signal = np.empty(p)
        for u in range(p):
            target = (2 * math.pi * u / p + self.phi) % (2 * math.pi)
            d = abs(angle_obs - target) % (2 * math.pi)
            d = min(d, 2 * math.pi - d)
            signal[u] = math.exp(-0.5 * (d / self._sigma) ** 2) / self._z_signal
        ells = (1.0 - eps) * signal + eps / (2.0 * math.pi)
        if self.phase == "CONF":
            print(f"   [dec conf t={self.t}] obs_angle={angle_obs:.3f}  "
                f"phi={self.phi:.3f}  "
                f"targets={[f'{(2*math.pi*u/p + self.phi) % (2*math.pi):.3f}' for u in range(p)]}  "
                f"ells={[f'{e:.3f}' for e in ells]}")
        return ells

    def _message_likelihood(self, ells: np.ndarray, belief: np.ndarray) -> np.ndarray:
        p = self.cfg.p_field
        M = len(belief)
        q = np.empty(M)
        cdf = np.concatenate([[0.0], np.cumsum(belief)])
        for j in range(M):
            lo, hi = cdf[j], cdf[j + 1]
            width = max(hi - lo, 1e-30)
            p_u = np.zeros(p)
            for u in range(p):
                u_lo, u_hi = u / p, (u + 1) / p
                ov = max(0.0, min(hi, u_hi) - max(lo, u_lo))
                p_u[u] = ov / width
            q[j] = float((p_u * ells).sum())
        return q

    def _posterior_match_symbol(self, belief: np.ndarray, m: int) -> int:
        R = float(self.rng.random())
        V = float(belief[:m].sum() + R * belief[m])
        p = self.cfg.p_field
        return min(int(p * V), p - 1)

    # ---- effective belief (with knockdowns) ----------------------------
    def effective_pi(self) -> np.ndarray:
        eff = self.pi * self.knockdown
        s = eff.sum()
        return eff / s if s > 0 else np.ones_like(eff) / len(eff)

    # ---- encoder step: produce one token ------------------------------
    def step_encoder(self, p_x, m_true):
        u = self._compute_codeword_symbol(m_true)
        print(f"   [enc step] t={self.t} t_offset={self.t_offset} phase={self.phase} u={u}")
        q = self.coupled_dist(p_x, self.t_offset + self.t, u)
        tok = self.sample_from(q)
        print(f"   [enc step] t={self.t} sampled tok={tok}")
        self._consume_token(tok)
        return tok

    # ---- decoder step: consume an observed token ----------------------
    def step_decoder(self, token_id):
        print(f"   [dec step] t={self.t} t_offset={self.t_offset} phase={self.phase} received tok={token_id}")
        self._consume_token(token_id)

    # ---- shared belief-update logic -----------------------------------
    def _compute_codeword_symbol(self, m_true: int) -> int:
        """Encoder-side: which u to send given current state."""
        if self.phase == "COMM":
            eff = self.effective_pi()
            return self._posterior_match_symbol(eff, m_true)
        else:  # CONF
            true_bit = 0 if self.candidate == m_true else 1
            return self._posterior_match_symbol(self.rho, true_bit)

    def _consume_token(self, token_id: int) -> None:
        if self.done:
            return
        angle_obs = self.angle_of_token(token_id, self.t_offset + self.t)
        ells = self._per_symbol_likelihood(angle_obs)
        self.t += 1

        if self.phase == "COMM":
            q = self._message_likelihood(ells, self.pi)
            self.pi = self.pi * q
            s = self.pi.sum()
            self.pi = self.pi / s if s > 0 else np.ones_like(self.pi) / self.M
            self.n_comm += 1
            eff = self.effective_pi()
            self._emit(EVT_COMM_UPDATE, {"top_idx": int(eff.argmax()),
                                         "top_prob": float(eff.max())})
            # check stops
            if eff.max() >= self.cfg.gamma_2:
                self.decoded = int(eff.argmax())
                self.done = True
                self._emit(EVT_GAMMA2_DIRECT, {"m": self.decoded, "pi": float(eff.max())})
                self._emit(EVT_DONE, {"m": self.decoded, "tokens": self.t, "via": "gamma2"})
            elif eff.max() >= self.cfg.gamma_1:
                self.candidate = int(eff.argmax())
                self.phase = "CONF"
                self.rho = np.array([0.5, 0.5])
                self._emit(EVT_GAMMA1_CROSS,
                           {"candidate": self.candidate, "pi": float(eff.max())})
        else:  # CONF
            q = self._message_likelihood(ells, self.rho)
            self.rho = self.rho * q
            self.rho = self.rho / self.rho.sum()
            self.n_conf += 1
            self._emit(EVT_CONF_UPDATE, {"rho_ack": float(self.rho[0]),
                                         "rho_nack": float(self.rho[1])})
            if self.rho[0] >= self.cfg.rho_ack:
                self.decoded = self.candidate
                self.done = True
                self._emit(EVT_ACK, {"m": self.decoded})
                self._emit(EVT_DONE, {"m": self.decoded, "tokens": self.t, "via": "ack"})
            elif self.rho[1] >= self.cfg.rho_nack:
                self._emit(EVT_NACK, {"candidate": self.candidate})
                # soft Bayes knockdown
                rn = self.cfg.rho_nack
                self.knockdown[self.candidate] *= (1 - rn) / rn
                # absorb knockdown into pi so future effective_pi reads clean
                self.pi = self.pi * self.knockdown
                self.pi = self.pi / self.pi.sum()
                self.knockdown = np.ones(self.M)
                self.phase = "COMM"
                self.candidate = None
                self.rho = None

    # ---- diagnostics ---------------------------------------------------
    def _emit(self, tag: str, payload: dict) -> None:
        if self.on_event is not None:
            payload = {**payload, "t": self.t, "phase": self.phase}
            self.on_event(tag, payload)

    def top_k(self, k: int = 3) -> list[tuple[int, float]]:
        eff = self.effective_pi()
        idx = np.argsort(eff)[::-1][:k]
        return [(int(i), float(eff[i])) for i in idx]