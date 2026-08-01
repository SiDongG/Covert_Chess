"""Adapter around ArcMark's encoding/decoding primitives.

  1. Channel coding: pick a random linear code over F_p with k×n generator G
     (shared between the two parties). The codeword is `C_m = m G  (mod p)`.
  2. At step t, form the channel-input angle
        z_t = (2π C_m(t)/p + 2π V_t/r + φ) mod 2π
     where V_t in {0,...,r-1} is a shared-secret nonce.
  3. Token sampling: draw X_t from the optimal-transport coupled distribution
        Q*_{X_t|Z_t} = argmin E[d(2π Π_t(X_t)/V, Z_t)]   s.t.  marginal = p_x_t.
     Π_t is the shared-secret per-step permutation of the vocabulary, V = |vocab|.
     This is solved with Sinkhorn over the (p × V) cost matrix.
  4. Decoding: from observed token x_t,
        Ĉ_ang(t) = (2π Π_t(x_t)/V - 2π V_t/r) mod 2π
     Choose
        m̂ = argmin_m  Σ_t f(d(Ĉ_ang(t), 2π C_m(t)/p + φ))
     Decoder needs ONLY the text + shared secret; no LM access required.

================================================================================
THIS FILE IS A WRAPPER AROUND ArcMark.
================================================================================
The OT-coupling (`_solve_ot_coupling`), token sampling (`sample_watermarked`),
and ML decoder (`decode_payload`) all defer to the reference ArcMark
implementation in :mod:`arcmark`. The shared secret (G, V, Π, φ) and the
linear-code construction remain local to this adapter so the rest of the
pipeline (encoder.py / decoder.py) sees the same surface as before.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from arcmark import (
    ArcMarkConfig as _ArcMarkSolverConfig,
    extract_conditional,
    score_all_messages,
    solve_arcmark_ot,
)


# ---------- LM interface that the adapter actually needs ----------

class LMBackendLike(Protocol):
    vocab_size: int

    def next_token_distribution(self, input_ids: torch.LongTensor) -> torch.Tensor: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


# ---------- config ----------

@dataclass
class ArcMarkConfig:
    k_bits: int = 16          # payload length (matches source_coding.PAYLOAD_BITS)
    n_tokens: int = 64        # tokens per payload
    p_field: int = 16          # codeword alphabet F_p
    r_resolution: int = 8     # nonce V_t alphabet
    shared_seed: int = 0xA12C # both parties derive G, {V_t}, {Π_t}, φ from this
    top_k: int | None = 50    # Sinkhorn vocabulary restriction (None disables)
    top_p: float | None = None  # optional nucleus restriction inside top-k
    # Sinkhorn solver knobs. ArcMark's own defaults (max_iter=500, stop_thr=1e-5)
    # are tuned for benchmarks, not LM-induced distributions which are often
    # peaky and ill-conditioned — Sinkhorn fails to converge surprisingly often
    # on them. We default to more headroom: more iterations, a looser
    # convergence target, slightly more entropy regularization.
    sinkhorn_max_iter: int = 4000
    sinkhorn_stop_thr: float = 1e-4
    sinkhorn_reg: float = 0.2
    sinkhorn_method: str = "sinkhorn_log"  # sinkhorn / sinkhorn_log / sinkhorn_stabilized


# ---------- adapter ----------

class ArcMarkAdapter:
    def __init__(self, lm: LMBackendLike, cfg: ArcMarkConfig | None = None):
        self.lm = lm
        self.cfg = cfg or ArcMarkConfig()
        self._init_shared_secret()
        # Lazy: built on first decode call.
        self._codebook_cache: torch.Tensor | None = None
        # Lazy: built on first encode call; ArcMark's Sinkhorn knobs.
        self._solver_cfg: _ArcMarkSolverConfig | None = None

    # ---- shared secret (deterministic from cfg.shared_seed) ----
    def _init_shared_secret(self) -> None:
        rng = np.random.default_rng(self.cfg.shared_seed)
        self.G = rng.integers(
            0, self.cfg.p_field, size=(self.cfg.k_bits, self.cfg.n_tokens), dtype=np.int64
        )
        self.V = rng.integers(0, self.cfg.r_resolution, size=self.cfg.n_tokens, dtype=np.int64)
        self.phi = float(rng.uniform(0.0, 2 * math.pi))
        self._perm_cache: dict[int, np.ndarray] = {}

    def _permutation(self, t: int) -> np.ndarray:
        if t not in self._perm_cache:
            seed_bytes = hashlib.blake2b(
                f"arcmark-perm:{self.cfg.shared_seed}:{t}".encode(), digest_size=8
            ).digest()
            seed = int.from_bytes(seed_bytes, "big")
            rng = np.random.default_rng(seed)
            self._perm_cache[t] = rng.permutation(self.lm.vocab_size)
        return self._perm_cache[t]

    def _permutation_torch(self, t: int) -> torch.Tensor:
        return torch.from_numpy(self._permutation(t).astype(np.int64))

    def _arcmark_solver_config(self) -> _ArcMarkSolverConfig:
        # Keys here are independent of arcmark_adapter.ArcMarkConfig — these
        # control the Sinkhorn solver, not the channel code. Hash-keying is
        # off because our shared secret already determines (V_t, Π_t).
        if self._solver_cfg is None:
            self._solver_cfg = _ArcMarkSolverConfig(
                hash_keys=False,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
                max_iter=self.cfg.sinkhorn_max_iter,
                stop_thr=self.cfg.sinkhorn_stop_thr,
                sinkhorn_reg=self.cfg.sinkhorn_reg,
                method=self.cfg.sinkhorn_method,
            )
        return self._solver_cfg

    # ---- channel coding ----
    def codeword(self, payload_bits: list[int]) -> np.ndarray:
        if len(payload_bits) != self.cfg.k_bits:
            raise ValueError(
                f"payload has {len(payload_bits)} bits, expected {self.cfg.k_bits}"
            )
        m = np.asarray(payload_bits, dtype=np.int64)
        return (m @ self.G) % self.cfg.p_field

    def channel_angle(self, c_t: int, t: int) -> float:
        return (
            2 * math.pi * c_t / self.cfg.p_field
            + 2 * math.pi * float(self.V[t]) / self.cfg.r_resolution
            + self.phi
        ) % (2 * math.pi)

    # =========================================================================
    # (A) OT coupling — defers to ArcMark's Sinkhorn (arcmark.sinkhorn).
    # Returns the length-V conditional P*(x | s = V[t]) after restricting
    # vocabulary by the solver's top-k.
    # =========================================================================
    def _solve_ot_coupling(self, p_xt: np.ndarray, t: int, c_t: int) -> np.ndarray:
        probs = torch.from_numpy(np.ascontiguousarray(p_xt)).to(torch.float64)
        perm_t = self._permutation_torch(t)
        s_index = int(self.V[t])

        try:
            ot_result = solve_arcmark_ot(
                probs,
                codeword_symbol=int(c_t),
                alphabet_size=self.cfg.p_field,
                num_keys=self.cfg.r_resolution,
                vocab_size=self.lm.vocab_size,
                perm=perm_t,
                phi=self.phi,
                config=self._arcmark_solver_config(),
            )
            cond = extract_conditional(
                ot_result.coupling,
                s_index,
                num_keys=self.cfg.r_resolution,
                full_vocab_size=self.lm.vocab_size,
                token_indices=ot_result.token_indices,
            )
        except Exception:
            # Sinkhorn occasionally diverges on ill-conditioned distributions;
            # fall back to the un-watermarked LM distribution for this token.
            return p_xt

        q = cond.cpu().numpy()
        if not np.all(np.isfinite(q)) or q.sum() <= 0:
            return p_xt
        q = np.clip(q, 0.0, None)
        q /= q.sum()
        return q
    
    def _solve_ot_coupling_array(self, p_xt_np, t: int, c_t: int):
        return self._solve_ot_coupling(p_xt_np, t, c_t)
    # =========================================================================
    # (B) Sample one watermarked token (used by the encoder, one token at a time)
    # =========================================================================
    def sample_watermarked(self, p_xt: torch.Tensor, t: int, c_t: int) -> int:
        if t < 0 or t >= self.cfg.n_tokens:
            raise IndexError(f"step t={t} out of range [0, {self.cfg.n_tokens})")
        p_np = p_xt.detach().to(torch.float64).cpu().numpy()
        q = self._solve_ot_coupling(p_np, t, c_t)
        return int(np.random.choice(self.lm.vocab_size, p=q))

    # ---- decoder helpers ----
    def _angle_of_observed_token(self, token_id: int, t: int) -> float:
        # Matches arcmark.symbol_decoder.decode_symbol_angle, but uses our
        # locally-cached numpy permutation rather than re-seeding torch.
        perm = self._permutation(t)
        return (
            2 * math.pi * float(perm[token_id]) / self.lm.vocab_size
            - 2 * math.pi * float(self.V[t]) / self.cfg.r_resolution
        ) % (2 * math.pi)

    def _codebook(self) -> torch.Tensor:
        """All 2^k_bits codewords as a (M, n_tokens) long tensor."""
        if self._codebook_cache is None:
            k = self.cfg.k_bits
            M = 1 << k
            # Build the (M, k) bit-vector matrix (MSB-first), then multiply by G.
            idx = np.arange(M, dtype=np.int64)[:, None]
            shifts = (k - 1 - np.arange(k, dtype=np.int64))[None, :]
            bits = (idx >> shifts) & 1                  # shape (M, k)
            codewords = (bits @ self.G) % self.cfg.p_field  # shape (M, n_tokens)
            self._codebook_cache = torch.from_numpy(codewords).long()
        return self._codebook_cache

    # =========================================================================
    # (C) Decoder — defers to ArcMark's `score_all_messages` with the
    # capacity-achieving "log" scoring function (Eq. (17) of the paper).
    # =========================================================================
    def decode_payload(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) != self.cfg.n_tokens:
            raise ValueError(
                f"need exactly {self.cfg.n_tokens} tokens to decode, got {len(token_ids)}"
            )

        angle_estimates = torch.tensor(
            [self._angle_of_observed_token(token_ids[t], t) for t in range(self.cfg.n_tokens)],
            dtype=torch.float64,
        )
        scores = score_all_messages(
            angle_estimates,
            self._codebook(),
            alphabet_size=self.cfg.p_field,
            phi=self.phi,
            scoring="log",
        )
        m_int = int(scores.argmin().item())
        return [(m_int >> (self.cfg.k_bits - 1 - i)) & 1 for i in range(self.cfg.k_bits)]

    # ---- pipeline conveniences ----
    @property
    def n_tokens_per_message(self) -> int:
        return self.cfg.n_tokens
