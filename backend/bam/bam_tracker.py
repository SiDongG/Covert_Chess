from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# --- event tags used by the optional debug callback ----------------------
EVT_COMM_UPDATE   = "comm_update"
EVT_GAMMA1_CROSS  = "gamma1_cross"
EVT_CONF_UPDATE   = "conf_update"
EVT_ACK           = "ack"
EVT_NACK          = "nack"
EVT_DONE          = "done"


@dataclass
class BAMConfig:
    # Mixture-likelihood contamination probabilities (in [0, 1)).
    # Values match the reference implementation (compare.py): EPS_NOISE=0.4
    # for the communication-phase Laplace floor, EPS_CONF=0.4 for the
    # confirmation-phase antipodal floor.
    eps_noise_comm: float = 0.4   # EPS_NOISE  (comm Laplace floor)
    eps_noise_conf: float = 0.4   # EPS_CONF   (conf antipodal floor)
    gamma_1: float = 0.5    # GAMMA — COMM → CONF threshold (g1)
    # rho_ACK: reference uses ra = 1 - 1/L where L is the message-alphabet size.
    # Here L = M (legal moves). Leave as None to auto-derive 1 - 1/M per message;
    # set an explicit float to override.
    rho_ack: Optional[float] = None
    rho_nack: float = 0.75  # RHO_NACK
    p_field: int = 4        # channel alphabet size |U|


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
        # LAPLACE signal kernel, matching compare.py exactly:
        #   sigma = pi / P_SYM ;  b = sigma / sqrt(2)
        #   Z_signal = 2 b (1 - exp(-pi / b))
        # (The previous code used a Gaussian kernel, which does not match the
        #  reference and shifts every posterior update.)
        p = self.cfg.p_field
        self._sigma = math.pi / p
        self._b_laplace = self._sigma / math.sqrt(2.0)
        self._z_signal = 2.0 * self._b_laplace * (1.0 - math.exp(-math.pi / self._b_laplace))
        # Confirmation phase: genuine 2-symbol ANTIPODAL channel.
        #   ACK  -> symbol 0     (angle phi)
        #   NACK -> symbol p//2  (angle phi + pi, antipodal)
        # reuse the comm Laplace scale, own normaliser, own floor (eps_noise_conf).
        self._sym_ack  = 0
        self._sym_nack = p // 2
        self._angle_ack  = (2 * math.pi * self._sym_ack  / p + self.phi) % (2 * math.pi)
        self._angle_nack = (2 * math.pi * self._sym_nack / p + self.phi) % (2 * math.pi)
        self._b_conf = self._b_laplace
        self._z_conf = 2.0 * self._b_conf * (1.0 - math.exp(-math.pi / self._b_conf))
        # rho_ACK: auto-derive 1 - 1/M when not explicitly set (reference ra=1-1/L).
        self._rho_ack = (self.cfg.rho_ack if self.cfg.rho_ack is not None
                         else (1.0 - 1.0 / self.M))

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
        p   = self.cfg.p_field
        eps = self.cfg.eps_noise_comm   # COMM-phase floor (Laplace)
        signal = np.empty(p)
        for u in range(p):
            target = (2 * math.pi * u / p + self.phi) % (2 * math.pi)
            d = abs(angle_obs - target) % (2 * math.pi)
            d = min(d, 2 * math.pi - d)
            signal[u] = math.exp(-d / self._b_laplace) / self._z_signal
        ells = (1.0 - eps) * signal + eps / (2.0 * math.pi)
        return ells

    def _conf_two_symbol_likelihood(self, angle_obs: float) -> np.ndarray:
        """Clean 2-hypothesis likelihood over the antipodal pair {ACK, NACK}.

        Matches compare.py::conf_two_symbol_likelihood — Laplace signal toward
        each of the TWO antipodal targets only (symbols 0 and p//2), mixed with
        the confirmation floor eps_noise_conf. This is the genuine confirmation
        channel; the old code reused the 4-symbol comm likelihood over a 2-vector,
        which is NOT the antipodal channel and inflated confirmation errors.
        """
        def circ(a, b):
            d = abs(a - b) % (2 * math.pi)
            return min(d, 2 * math.pi - d)
        d_ack  = circ(angle_obs, self._angle_ack)
        d_nack = circ(angle_obs, self._angle_nack)
        eps = self.cfg.eps_noise_conf
        sig_ack  = math.exp(-d_ack  / self._b_conf) / self._z_conf
        sig_nack = math.exp(-d_nack / self._b_conf) / self._z_conf
        ell_ack  = (1.0 - eps) * sig_ack  + eps / (2.0 * math.pi)
        ell_nack = (1.0 - eps) * sig_nack + eps / (2.0 * math.pi)
        return np.array([ell_ack, ell_nack])

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
        q = self.coupled_dist(p_x, self.t_offset + self.t, u)
        tok = self.sample_from(q)
        self._consume_token(tok)
        return tok

    # ---- decoder step: consume an observed token ----------------------
    def step_decoder(self, token_id):
        self._consume_token(token_id)

    # ---- shared belief-update logic -----------------------------------
    def _compute_codeword_symbol(self, m_true: int) -> int:
        """Encoder-side: which u to send given current state."""
        if self.phase == "COMM":
            eff = self.effective_pi()
            return self._posterior_match_symbol(eff, m_true)
        else:  # CONF — genuine antipodal channel: emit the symbol DIRECTLY by
               # the bit (no CDF/posterior-match over the 2-belief). This matches
               # compare.py::run_confirmation (tx_symbol = SYM_ACK|SYM_NACK).
            true_bit = 0 if self.candidate == m_true else 1
            return self._sym_ack if true_bit == 0 else self._sym_nack

    def _consume_token(self, token_id: int) -> None:
        if self.done:
            return
        angle_obs = self.angle_of_token(token_id, self.t_offset + self.t)
        self.t += 1

        if self.phase == "COMM":
            ells = self._per_symbol_likelihood(angle_obs)
            q = self._message_likelihood(ells, self.pi)
            self.pi = self.pi * q
            s = self.pi.sum()
            self.pi = self.pi / s if s > 0 else np.ones_like(self.pi) / self.M
            self.n_comm += 1
            eff = self.effective_pi()
            self._emit(EVT_COMM_UPDATE, {"top_idx": int(eff.argmax()),
                                         "top_prob": float(eff.max())})
            # Only one threshold: once gamma_1 is crossed, always enter CONF.
            # There is no direct-decode shortcut — every decision must be
            # confirmed by the Yamamoto-Itoh confirmation phase.
            if eff.max() >= self.cfg.gamma_1:
                self.candidate = int(eff.argmax())
                self.phase = "CONF"
                self.rho = np.array([0.5, 0.5])
                self._emit(EVT_GAMMA1_CROSS,
                           {"candidate": self.candidate, "pi": float(eff.max())})
        else:  # CONF — clean 2-symbol antipodal likelihood on rho=[P(ACK),P(NACK)]
            ell = self._conf_two_symbol_likelihood(angle_obs)
            self.rho = self.rho * ell
            self.rho = self.rho / self.rho.sum()
            self.n_conf += 1
            self._emit(EVT_CONF_UPDATE, {"rho_ack": float(self.rho[0]),
                                         "rho_nack": float(self.rho[1])})
            if self.rho[0] >= self._rho_ack:
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