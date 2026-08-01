"""GF(2^k) multiplicative codebook for ArcMark multi-bit watermarking.

This module provides :class:`GFCode` — a drop-in replacement for
:class:`~arcmark.coding.RandomLinearCode` that uses a Reed-Solomon-like
structure over GF(2^k) instead of a random linear code over Z_p.

**Why GF(2^k) is better for short codewords (n ≈ 300):**

RandomLinearCode over Z_p offers no minimum-distance guarantee for finite
n.  Two codewords can collide at many positions by chance, reducing the
effective signal per token.

GFCode uses the geometric sequence:

    codeword[m, t] = m · g^t  in GF(2^k)

where g is the primitive element of GF(2^k) and multiplication is the
field multiplication.  This has two provably optimal properties:

1. **Maximum separation:** For any two distinct messages m ≠ m',
   m·g^t ≠ m'·g^t at EVERY position t (field multiplication by a
   nonzero element is bijective).  No two codewords share any symbol.

2. **Uniform distribution:** As t varies, m·g^t cycles through all
   2^k - 1 nonzero elements exactly once per period 2^k - 1.
   Symbols are perfectly spread across the alphabet.

This is essentially a Reed-Solomon code — an MDS code achieving the
Singleton bound, optimal for any block length.

**Supported k:** 2–16 (built-in primitive polynomials).
For k > 16 the codebook (2^k × n) becomes too large to store;
use :class:`~arcmark.efficient_random_linear_code.EfficientRandomLinearCode`
for those cases.

**Usage:**

    code = GFCode.build(k_bits=8, codeword_length=300)
    codeword = code.encode(42)      # shape (300,), values in {0,...,255}
    codebook  = code.codebook       # shape (256, 300)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from arcmark.coding import ChannelCode

__all__ = ["GFCode"]


# ─────────────────────────────────────────────────────────────────────────────
# Primitive polynomials for GF(2^k)
# ─────────────────────────────────────────────────────────────────────────────

# Monic primitive polynomials over GF(2) — the high bit (x^k) is implicit.
# poly & ((1<<k)-1) gives the lower k coefficients.
_PRIM_POLY: dict[int, int] = {
    2:  0b111,       # x^2 + x + 1
    3:  0b1011,      # x^3 + x + 1
    4:  0b10011,     # x^4 + x + 1
    5:  0b100101,    # x^5 + x^2 + 1
    6:  0b1000011,   # x^6 + x + 1
    7:  0b10000011,  # x^7 + x + 1
    8:  0x11D,       # x^8 + x^4 + x^3 + x^2 + 1
    9:  0x211,       # x^9 + x^4 + 1
    10: 0x409,       # x^10 + x^3 + 1
    11: 0x805,       # x^11 + x^2 + 1
    12: 0x1053,      # x^12 + x^3 + x + 1
    13: 0x201B,      # x^13 + x^4 + x^3 + x + 1
    14: 0x4003,      # x^14 + x + 1
    15: 0x8003,      # x^15 + x + 1
    16: 0x1100B,     # x^16 + x^12 + x^3 + x + 1
}


def _build_gf_tables(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Build exp and log tables for GF(2^k).

    Returns:
        exp: shape (2*(2^k-1),) int32 — exp[i] = g^i, duplicated for wrap
        log: shape (2^k,)      int32 — log[x] = i s.t. g^i = x; log[0] = -1
    """
    if k not in _PRIM_POLY:
        raise ValueError(
            f"No built-in primitive polynomial for k={k}. "
            f"Supported: {sorted(_PRIM_POLY.keys())}"
        )
    prim = _PRIM_POLY[k]
    m    = 1 << k
    mask = m - 1
    period = m - 1   # multiplicative group order

    exp = np.zeros(2 * period, dtype=np.int32)
    log = np.full(m, -1, dtype=np.int32)

    x = 1
    for i in range(period):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & m:
            x ^= prim
        x &= mask

    exp[period:] = exp[:period]   # duplicate for modular wrap
    return exp, log


def _build_codebook(k: int, n: int,
                    exp: np.ndarray, log: np.ndarray) -> np.ndarray:
    """Build GF(2^k) codebook: codebook[m, t] = m * g^t in GF(2^k).

    Args:
        k:   Number of bits (GF alphabet = 2^k).
        n:   Codeword length.
        exp: Exponent table from _build_gf_tables.
        log: Logarithm table from _build_gf_tables.

    Returns:
        int32 array of shape (2^k, n), values in {0,...,2^k-1}.
    """
    M      = 1 << k
    period = M - 1
    cb     = np.zeros((M, n), dtype=np.int32)

    # t_exp[t] = t mod period  (the exponent for g^t)
    t_exp = np.arange(n, dtype=np.int64) % period

    for m in range(1, M):
        e_m        = int(log[m])
        row_exp    = (e_m + t_exp) % period
        cb[m, :]   = exp[row_exp]

    # m=0 stays all-zero (0 * anything = 0 in GF)
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# GFCode class
# ─────────────────────────────────────────────────────────────────────────────

class GFCode(ChannelCode):
    """Reed-Solomon-like code over GF(2^k) for ArcMark watermarking.

    Codeword for message m: ``[m, m·g, m·g², ..., m·g^{n-1}]`` in GF(2^k).

    Optimal minimum distance: any two distinct codewords differ at ALL n
    positions (since GF multiplication by nonzero is bijective).

    Args:
        k_bits:          Number of information bits k (alphabet = 2^k).
        codeword_length: Number of watermarked tokens n.

    Example::

        code = GFCode.build(k_bits=8, codeword_length=300)
        cw   = code.encode(42)      # shape (300,)
        cb   = code.codebook        # shape (256, 300)
    """

    def __init__(
        self,
        k_bits: int,
        codeword_length: int,
        _codebook: np.ndarray,   # (M, n) int32
        _exp: np.ndarray,
        _log: np.ndarray,
    ) -> None:
        self._k            = k_bits
        self._n            = codeword_length
        self._M            = 1 << k_bits
        self._p            = 1 << k_bits   # alphabet size = 2^k
        self._cb_np        = _codebook     # (M, n) numpy for fast encoding
        self._cb_torch     = torch.from_numpy(_codebook).long()
        self._exp          = _exp
        self._log          = _log

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        k_bits: int,
        codeword_length: int,
    ) -> GFCode:
        """Build a GF(2^k) code.

        Args:
            k_bits:          Bits per message (2–16).
            codeword_length: Number of watermarked tokens.

        Returns:
            Initialised GFCode.
        """
        if k_bits < 2 or k_bits > 16:
            raise ValueError(
                f"GFCode supports k_bits in 2..16, got {k_bits}. "
                f"For k>16 use EfficientRandomLinearCode."
            )
        exp, log = _build_gf_tables(k_bits)
        cb       = _build_codebook(k_bits, codeword_length, exp, log)
        return cls(
            k_bits=k_bits,
            codeword_length=codeword_length,
            _codebook=cb,
            _exp=exp,
            _log=log,
        )

    # ── ChannelCode abstract implementations ─────────────────────────────

    def encode(self, message_idx: int) -> Tensor:
        if not (0 <= message_idx < self._M):
            raise IndexError(
                f"message_idx={message_idx} out of range [0, {self._M})"
            )
        return self._cb_torch[message_idx]

    @property
    def codebook(self) -> Tensor:
        return self._cb_torch

    @property
    def num_messages(self) -> int:
        return self._M

    @property
    def codeword_length(self) -> int:
        return self._n

    @property
    def alphabet_size(self) -> int:
        return self._p

    @property
    def k_bits(self) -> int:
        return self._k
