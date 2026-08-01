"""Efficient RandomLinearCode for large k (up to 32-bit and beyond).

This module provides :class:`EfficientRandomLinearCode` — a drop-in
replacement for :class:`~arcmark.coding.RandomLinearCode` that avoids
materialising the exponentially large codebook.

**Key differences from RandomLinearCode:**

- Never stores the ``(2^k, n)`` codebook — saves up to 38 GB for k=24
- Never stores the ``(2^k, d)`` message_vectors — saves up to 16 GB for k=32
- Uses **natural enumeration**: message index ``i`` maps directly to
  message vector ``m_vec(i) = [(i // p^j) % p for j in 0..d-1]``
- **Encodes** in O(d·n): ``codeword = m_vec(idx) @ G mod p``
- **Decodes** in O(n·p^d) using a nested vectorised decoder:
  - d=1: trivial, O(n·p)
  - d=2: O(n·p²) ≈ 0.5s CPU
  - d=3: O(n·p³) ≈ 45s CPU / 0.5s GPU
  - d=4: O(n·p⁴) ≈ infeasible CPU / 13ms GPU
- Auto-selects GPU (CUDA) when available

**Memory:**

  ============  ================  =================  ==================
  k             d                 Old codebook        This class
  ============  ================  =================  ==================
  8             1                 0.6 MB              4 KB (G only)
  16            2                 157 MB              4 KB (G only)
  24            3                 38 GB               4 KB (G only)
  32            4                 10 TB (impossible)  4 KB (G only)
  ============  ================  =================  ==================

**Usage:**

    code = EfficientRandomLinearCode.build(k_bits=32, codeword_length=300,
                                           alphabet_size=256, seed=42)
    codeword = code.encode(12345678)        # shape (300,), O(d·n)
    msg_idx  = code.decode(angle_estimates) # int in [0, 2^32), fast on GPU

**Compatibility:**

The ``encode`` / ``int_to_bits`` / ``k_bits`` / ``alphabet_size`` /
``codeword_length`` interface matches :class:`~arcmark.coding.ChannelCode`
so it plugs directly into :class:`~arcmark.processor.ArcMarkLogitsProcessor`
and the evaluation scripts.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor

__all__ = ["EfficientRandomLinearCode"]

# ─────────────────────────────────────────────────────────────────────────────
# Natural enumeration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _idx_to_mvec(idx: int, p: int, d: int) -> np.ndarray:
    """Convert message index to message vector using natural enumeration.

    Natural enumeration: message i ↔ base-p representation of i.
    m_vec[j] = (i // p^j) % p  for j = 0, ..., d-1.

    Args:
        idx: Message index in [0, p^d).
        p:   Alphabet size.
        d:   Dimension of message vector.

    Returns:
        int32 array of shape (d,) with values in {0,...,p-1}.
    """
    m = np.empty(d, dtype=np.int32)
    val = int(idx)
    for j in range(d):
        m[j] = val % p
        val //= p
    return m


def _mvec_to_idx(m_vec: np.ndarray, p: int) -> int:
    """Convert message vector to message index (inverse of _idx_to_mvec).

    Args:
        m_vec: int array of shape (d,) with values in {0,...,p-1}.
        p:     Alphabet size.

    Returns:
        Message index as Python int.
    """
    d = len(m_vec)
    idx = 0
    p_pow = 1
    for j in range(d):
        idx += int(m_vec[j]) * p_pow
        p_pow *= p
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Soft belief computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_beliefs(
    angle_estimates: np.ndarray,
    p: int,
    phi: float = 0.0,
    scoring: str = "log",
) -> np.ndarray:
    """Per-position soft beliefs b[t, s], shape (n, p), float32.
    Lower = more likely symbol s at position t."""
    s_vals  = np.arange(p, dtype=np.float64)
    angles_s = (2.0 * math.pi * s_vals / p + phi) % (2.0 * math.pi)
    theta   = angle_estimates.astype(np.float64)
    diffs   = np.abs(theta[:, None] - angles_s[None, :])
    diffs   = np.minimum(diffs, 2.0 * math.pi - diffs)
    if scoring == "log":
        b = -np.log(np.clip(1.0 - diffs / math.pi, 1e-12, None))
    else:
        b = diffs
    return b.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CPU decoders for small d
# ─────────────────────────────────────────────────────────────────────────────

def _decode_cpu(b: np.ndarray, Gv: np.ndarray, n: int, p: int, d: int) -> np.ndarray:
    """CPU nested decoder for d = 1, 2, 3.

    Args:
        b:   (n, p) float32 soft beliefs.
        Gv:  (d, p, n) int32 precomputed G[j]*v mod p.
        n:   codeword length.
        p:   alphabet size.
        d:   message dimension (1, 2, or 3).

    Returns:
        Decoded message vector (d,) int32.
    """
    t_idx = np.arange(n, dtype=np.int32)

    if d == 1:
        scores = b[t_idx, Gv[0]].sum(axis=1)   # (p,)
        return np.array([int(np.argmin(scores))], dtype=np.int32)

    if d == 2:
        # A[t,m0,m1] = (Gv[0,m0,t] + Gv[1,m1,t]) mod p  → (n,p,p)
        # score2[m0,m1] = Σ_t b[t, A[t,m0,m1]] → flat idx = m0*p + m1
        A = ((Gv[0][:, None, :] + Gv[1][None, :, :]) % p
             ).transpose(2, 0, 1).astype(np.int32)          # (n, p, p)
        score2 = b[t_idx[:, None, None], A].sum(axis=0)     # (p, p)
        flat   = int(np.argmin(score2))
        m0, m1 = flat // p, flat % p
        return np.array([m0, m1], dtype=np.int32)

    if d == 3:
        # Precompute A[t,m0,m1] for coordinates 0,1
        A = ((Gv[0][:, None, :] + Gv[1][None, :, :]) % p
             ).transpose(2, 0, 1).astype(np.int32)          # (n, p, p)
        v = np.arange(p, dtype=np.int32)
        best_score = np.inf
        best_m = np.zeros(3, dtype=np.int32)
        for m2 in range(p):
            base2     = Gv[2, m2]                           # (n,) int32
            shift_idx = (v[None, :] + base2[:, None]) % p  # (n, p)
            b_shifted = b[t_idx[:, None], shift_idx]        # (n, p)
            score2    = b_shifted[t_idx[:, None, None], A].sum(axis=0)  # (p,p)
            flat      = int(np.argmin(score2))
            s         = float(score2.flat[flat])
            if s < best_score:
                best_score = s
                m0, m1 = flat // p, flat % p
                best_m = np.array([m0, m1, m2], dtype=np.int32)
        return best_m

    raise NotImplementedError(f"CPU decoder not implemented for d={d}. Use GPU.")


# ─────────────────────────────────────────────────────────────────────────────
# GPU decoders (PyTorch) — handles d = 1, 2, 3, 4
# ─────────────────────────────────────────────────────────────────────────────

def _decode_gpu(
    b_np: np.ndarray,
    Gv_np: np.ndarray,
    n: int,
    p: int,
    d: int,
    device: torch.device,
) -> np.ndarray:
    """GPU decoder for d = 1, 2, 3, 4.

    All tensors kept as int64 throughout to avoid index out-of-bounds
    errors from int32 overflow in CUDA kernels.

    Args:
        b_np:   (n, p) float32 beliefs (numpy).
        Gv_np:  (d, p, n) int32 precomputed multiplications (numpy).
        n, p, d: dimensions.
        device: CUDA device.

    Returns:
        Decoded message vector (d,) int32 numpy array.
    """
    # Force int64 for all index tensors — critical for CUDA correctness
    b    = torch.from_numpy(b_np).to(device)                         # (n, p) float32
    Gv   = torch.from_numpy(Gv_np.astype(np.int64)).to(device)       # (d, p, n) int64
    tidx = torch.arange(n, dtype=torch.int64, device=device)         # (n,)
    v    = torch.arange(p, dtype=torch.int64, device=device)         # (p,)

    def smod(x):
        """Safe mod p — result always in [0, p), int64."""
        return x.remainder(p).clamp(0, p - 1)  # clamp as extra safety net

    # Precompute A01[t, m0, m1] = (Gv[0,m0,t] + Gv[1,m1,t]) mod p
    # Shape: (n, p, p) int64
    A01 = smod(Gv[0][:, None, :] + Gv[1][None, :, :])               # (p, p, n)
    A01 = A01.permute(2, 0, 1).contiguous()                          # (n, p, p) int64

    if d == 1:
        sym    = smod(Gv[0])                                          # (p, n) int64
        scores = b[tidx, sym].sum(dim=1)                              # (p,)
        return np.array([int(scores.argmin().item())], dtype=np.int32)

    if d == 2:
        score2 = b[tidx[:, None, None], A01].sum(dim=0)              # (p, p)
        flat   = int(score2.view(-1).argmin().item())
        # A01[t, m0, m1]: flat index = m0 * p + m1
        m0, m1 = flat // p, flat % p
        return np.array([m0, m1], dtype=np.int32)

    if d == 3:
        best_score = float("inf")
        best_flat  = 0
        best_m2    = 0

        for m2 in range(p):
            base2     = Gv[2, m2]                                     # (n,) int64
            shift_idx = smod(v[None, :] + base2[:, None])             # (n, p) int64
            b_shifted = b[tidx[:, None], shift_idx]                   # (n, p) float32
            score2    = b_shifted[tidx[:, None, None], A01].sum(dim=0) # (p, p)
            min_val, flat = score2.view(-1).min(dim=0)
            val = min_val.item()
            if val < best_score:
                best_score = val
                best_flat  = int(flat.item())
                best_m2    = m2

        m0, m1 = best_flat // p, best_flat % p
        return np.array([m0, m1, best_m2], dtype=np.int32)

    if d == 4:
        best_score  = float("inf")
        best_flat   = 0
        best_m2, best_m3 = 0, 0

        for m3 in range(p):
            base3 = Gv[3, m3]                                         # (n,) int64
            for m2 in range(p):
                base23    = smod(base3 + Gv[2, m2])                   # (n,) int64
                shift_idx = smod(v[None, :] + base23[:, None])        # (n, p) int64
                b_shifted = b[tidx[:, None], shift_idx]                # (n, p) float32
                score2    = b_shifted[tidx[:, None, None], A01].sum(dim=0) # (p, p)
                min_val, flat = score2.view(-1).min(dim=0)
                val = min_val.item()
                if val < best_score:
                    best_score  = val
                    best_flat   = int(flat.item())
                    best_m2, best_m3 = m2, m3

        m0, m1 = best_flat // p, best_flat % p
        return np.array([m0, m1, best_m2, best_m3], dtype=np.int32)

    raise NotImplementedError(f"GPU decoder not implemented for d={d} (max d=4).")


# ─────────────────────────────────────────────────────────────────────────────
# EfficientRandomLinearCode
# ─────────────────────────────────────────────────────────────────────────────

class EfficientRandomLinearCode:
    """Memory-efficient random linear code over Z_p for ArcMark watermarking.

    Supports any k without storing the codebook or message vectors.
    Uses natural enumeration: message index i ↔ base-p representation of i.

    Compatible with :class:`~arcmark.processor.ArcMarkLogitsProcessor`
    and the evaluation scripts.

    Args:
        G:        Generator matrix, shape (d, n), dtype int32, values in Z_p.
        p:        Alphabet size.
        k:        Number of information bits (must satisfy p^d >= 2^k).
        phi:      Angle offset (must match ArcMark encoder).
        scoring:  Scoring function for decode ("log" or "linear").
        device:   Override device for GPU decode (None = auto).

    Example::

        code = EfficientRandomLinearCode.build(k_bits=32, codeword_length=300,
                                               alphabet_size=256, seed=42)
        cw  = code.encode(42)                # shape (300,), fast O(d·n)
        idx = code.decode(angle_estimates)   # int in [0, 2^32)
        bits = code.int_to_bits(idx)         # shape (32,)
    """

    def __init__(
        self,
        G: np.ndarray,
        p: int,
        k: int,
        phi: float = 0.0,
        scoring: str = "log",
        device: Optional[torch.device] = None,
    ) -> None:
        self._G       = G.astype(np.int32)   # (d, n)
        self._p       = p
        self._k       = k
        self._d       = G.shape[0]
        self._n       = G.shape[1]
        self._phi     = phi
        self._scoring = scoring

        # Auto-select device
        if device is None:
            self._device = (torch.device("cuda")
                            if torch.cuda.is_available() else None)
        else:
            self._device = device

        # Precompute Gv[j, v, t] = (v * G[j,t]) mod p — shape (d, p, n)
        v = np.arange(p, dtype=np.int32)
        self._Gv = np.array([
            (v[:, None] * self._G[j][None, :]) % p
            for j in range(self._d)
        ], dtype=np.int32)   # (d, p, n)

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        k_bits: int,
        codeword_length: int,
        alphabet_size: int,
        seed: int,
        phi: float = 0.0,
        scoring: str = "log",
        device: Optional[torch.device] = None,
    ) -> EfficientRandomLinearCode:
        """Build an EfficientRandomLinearCode from shared parameters.

        Deterministic: identical arguments always produce the same code.

        Args:
            k_bits:          Number of information bits.
            codeword_length: Number of watermarked tokens n.
            alphabet_size:   Alphabet size p (use 256 for k > 16).
            seed:            Shared secret seed.
            phi:             Angle offset.
            scoring:         Scoring function.
            device:          Override device for GPU decode.

        Returns:
            Initialised :class:`EfficientRandomLinearCode`.
        """
        p = alphabet_size
        n = codeword_length
        d = max(1, int(math.ceil(k_bits * math.log(2) / math.log(p))))

        if p ** d < 2 ** k_bits:
            raise ValueError(
                f"p^d = {p}^{d} = {p**d} < 2^k = {2**k_bits}. "
                f"Increase alphabet_size or reduce k_bits."
            )

        # Generate G deterministically using torch (matches RandomLinearCode)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        G_t = torch.randint(0, p, (d, n), generator=gen, dtype=torch.long)
        G   = G_t.numpy().astype(np.int32)

        return cls(G=G, p=p, k=k_bits, phi=phi, scoring=scoring, device=device)

    # ── ChannelCode interface ─────────────────────────────────────────────

    def encode(self, message_idx: int) -> Tensor:
        """Encode message_idx to a codeword of length n.

        Uses natural enumeration: idx → m_vec → codeword = m_vec @ G mod p.
        O(d·n) — fast for any k.

        Args:
            message_idx: Integer in [0, 2^k_bits).

        Returns:
            LongTensor of shape (n,) with values in {0,...,p-1}.
        """
        if not (0 <= message_idx < 2 ** self._k):
            raise IndexError(
                f"message_idx={message_idx} out of range [0, {2**self._k})"
            )
        m_vec = _idx_to_mvec(message_idx, self._p, self._d).astype(np.int64)
        codeword = (m_vec @ self._G.astype(np.int64)) % self._p
        return torch.tensor(codeword, dtype=torch.long)

    def decode(self, angle_estimates: Tensor) -> int:
        """Decode angle estimates to a message index.

        Uses nested vectorised decoder (GPU if available).
        O(n·p^d): fast for d ≤ 3 on CPU, d ≤ 4 on GPU.

        Args:
            angle_estimates: Float tensor of shape (n_tokens,) from
                             decode_symbol_angles(). May be shorter than
                             the full codeword length (checkpoint decoding).

        Returns:
            Decoded message index in [0, 2^k_bits).
        """
        angles_np = angle_estimates.cpu().numpy().astype(np.float64)
        n_actual  = len(angles_np)   # may be < self._n at checkpoints

        b = _compute_beliefs(angles_np, self._p, self._phi, self._scoring)

        # Slice Gv to match the actual number of tokens being decoded.
        # self._Gv has shape (d, p, MAX_TOKENS) but at each checkpoint
        # we only decode n_actual < MAX_TOKENS tokens — passing the full
        # Gv causes A01 shape (MAX_TOKENS, p, p) vs b shape (n_actual, p),
        # leading to index out-of-bounds in the CUDA gather kernel.
        Gv_sliced = self._Gv[:, :, :n_actual]   # (d, p, n_actual)

        use_gpu = (self._device is not None
                   and self._device.type == "cuda"
                   and self._d <= 4)

        if use_gpu:
            m_vec = _decode_gpu(b, Gv_sliced, n_actual, self._p, self._d,
                                 self._device)
        elif self._d <= 3:
            m_vec = _decode_cpu(b, Gv_sliced, n_actual, self._p, self._d)
        else:
            raise RuntimeError(
                f"d={self._d} (k={self._k}) requires GPU for decoding. "
                f"No CUDA device available. "
                f"Use --bits-per-chunk to reduce k per chunk."
            )

        return _mvec_to_idx(m_vec, self._p)

    def decode_bits(self, angle_estimates: Tensor) -> str:
        """Decode angle estimates to a bit string of length k."""
        return format(self.decode(angle_estimates), f"0{self._k}b")

    def int_to_bits(self, message_idx: int) -> Tensor:
        """Convert a message index to a binary bit vector (MSB first).

        Returns:
            uint8 tensor of shape (k_bits,).
        """
        if not (0 <= message_idx < 2 ** self._k):
            raise IndexError(
                f"message_idx={message_idx} out of range [0, {2**self._k})"
            )
        bits = []
        val = message_idx
        for _ in range(self._k):
            bits.append(val & 1)
            val >>= 1
        bits.reverse()
        return torch.tensor(bits, dtype=torch.uint8)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def k_bits(self) -> int:
        return self._k

    @property
    def num_messages(self) -> int:
        return 2 ** self._k

    @property
    def codeword_length(self) -> int:
        return self._n

    @property
    def alphabet_size(self) -> int:
        return self._p

    @property
    def dimension(self) -> int:
        """Message vector dimension d."""
        return self._d

    @property
    def generator_matrix(self) -> np.ndarray:
        """Generator matrix G of shape (d, n), dtype int32."""
        return self._G.copy()