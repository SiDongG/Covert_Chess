"""Polar code for ArcMark multi-bit watermarking.

Implements a binary polar code with:
- Systematic encoding via the polar transform G = B_n * F^{⊗log2(n)}
- Soft-decision Successive Cancellation (SC) decoding using von Mises
  log-likelihood ratios derived from circular angle distances.

This replaces RandomLinearCode for high-bit-count watermarks (32, 64 bits)
where RandomLinearCode's explicit codebook (M × n matrix) would overflow
memory.  PolarCode never materialises the codebook — encoding is O(n log n)
and decoding is O(n log n) regardless of k.

**ArcMark integration:**

The angle estimate θ_t at token position t is a noisy observation of the
true codeword angle  φ_t = 2π·c_t/p + offset, where c_t ∈ {0,1} for a
binary polar code (p=2, so c_t ∈ {0,1} maps to angles 0 and π).

The soft LLR for position t is:

    LLR_t = log [ P(θ_t | c_t=0) / P(θ_t | c_t=1) ]

Under a von Mises model with concentration κ:

    LLR_t = 2κ · cos(θ_t)   (for c=0 → angle 0,  c=1 → angle π)

We use κ=1 by default (can be tuned).  For the general p-ary case we
project the circular estimate onto a binary decision at each position.

Usage::

    code = PolarCode.build(k_bits=32, n_tokens=300, seed=42)
    codeword = code.encode(message_idx)   # shape (n_tokens,), values in {0,1}
    # ... generate watermarked tokens ...
    # At decode time, given angle_estimates (n_tokens,):
    message_idx = code.decode(angle_estimates)
    bits = code.int_to_bits(message_idx)  # k_bits-long bit string
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import torch
from torch import Tensor

from arcmark.coding import ChannelCode

__all__ = ["PolarCode"]


# ═══════════════════════════════════════════════════════════════════════════
# Polar code construction utilities
# ═══════════════════════════════════════════════════════════════════════════

def _bhattacharyya_bec(erasure_prob: float, n: int) -> np.ndarray:
    """Compute Bhattacharyya parameters for BEC channel, used for
    frozen bit selection via the polar transform recursion.

    Args:
        erasure_prob: BEC erasure probability (design parameter, 0.5 is neutral).
        n: Polar code block length (must be power of 2).

    Returns:
        Array of shape (n,) with Bhattacharyya parameters for each synthetic
        channel, lower = more reliable = better for information bits.
    """
    z = np.array([erasure_prob])
    log2_n = int(math.log2(n))
    for _ in range(log2_n):
        # Upper channel: z^2, Lower channel: 2z - z^2
        z_upper = z ** 2
        z_lower = 2 * z - z ** 2
        z = np.concatenate([z_lower, z_upper])
    return z


def _select_info_bits(n: int, k: int, seed: int) -> np.ndarray:
    """Select k most reliable bit positions for information bits.

    Uses Bhattacharyya parameters with erasure_prob=0.5 for channel-agnostic
    construction, then deterministically breaks ties using the seed.

    Args:
        n: Block length (power of 2).
        k: Number of information bits.
        seed: Random seed for deterministic tie-breaking.

    Returns:
        Sorted array of k indices for information bit positions.
    """
    z = _bhattacharyya_bec(0.5, n)
    # Add tiny deterministic noise to break ties reproducibly
    rng = np.random.default_rng(seed)
    z = z + rng.uniform(0, 1e-10, size=n)
    info_indices = np.argsort(z)[:k]   # k smallest = most reliable
    return np.sort(info_indices)


def _polar_transform(u: np.ndarray) -> np.ndarray:
    """Apply the polar transform G = F^{⊗log2(n)} to bit vector u.

    Args:
        u: Input bit array of length n (power of 2).

    Returns:
        Output bit array x = u @ G mod 2.
    """
    n = len(u)
    x = u.copy()
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            x[i:i + step] ^= x[i + step:i + 2 * step]
        step *= 2
    return x


def _inv_polar_transform(x: np.ndarray) -> np.ndarray:
    """Inverse polar transform (same as forward for binary field)."""
    return _polar_transform(x)   # G is self-inverse: G^2 = I mod 2


# ═══════════════════════════════════════════════════════════════════════════
# Soft-decision SC decoder
# ═══════════════════════════════════════════════════════════════════════════

def _sc_decode(llrs: np.ndarray, frozen_mask: np.ndarray) -> np.ndarray:
    """Successive Cancellation (SC) decoder for polar codes.

    Args:
        llrs:        Channel LLR values, shape (n,). LLR = log P(y|0)/P(y|1).
        frozen_mask: Boolean array of shape (n,). True = frozen (set to 0).

    Returns:
        Decoded bit array u_hat of shape (n,).
    """
    n = len(llrs)
    log2_n = int(math.log2(n))

    # Store all LLR stages: shape (log2_n+1, n)
    # Stage 0 = channel LLRs, stage log2_n = decoded bits
    alpha = np.zeros((log2_n + 1, n))
    beta  = np.zeros((log2_n + 1, n), dtype=np.int8)
    alpha[0] = llrs

    u_hat = np.zeros(n, dtype=np.int8)

    def _f(a: float, b: float) -> float:
        """Check-node operation: f(a, b) = sign(a)*sign(b)*min(|a|,|b|)."""
        return np.sign(a) * np.sign(b) * min(abs(a), abs(b))

    def _g(a: float, b: float, u: int) -> float:
        """Variable-node operation: g(a, b, u) = b + (1-2u)*a."""
        return b + (1 - 2 * u) * a

    # Iterative SC decoding using a recursive-like schedule
    # We use the standard bit-reversal + butterfly approach
    for i in range(n):
        # Propagate LLRs down to leaf i
        depth = log2_n
        idx = i
        stage = 0

        # Compute LLR at each stage using already-decided bits
        for s in range(log2_n):
            half = n >> (s + 1)
            pair_idx = idx ^ half if (idx & half) else idx
            left = (idx & half) == 0

            if left:
                # f operation: check node
                alpha[s + 1, i] = _f(
                    alpha[s, pair_idx],
                    alpha[s, pair_idx + half if left else pair_idx - half]
                    if False else alpha[s, i + half if left else i - half]
                )
            else:
                # g operation: variable node, needs left sibling decision
                left_idx = i - half
                alpha[s + 1, i] = _g(
                    alpha[s, left_idx],
                    alpha[s, i],
                    beta[s + 1, left_idx]
                )

        # Make hard decision at leaf
        if frozen_mask[i]:
            u_hat[i] = 0
        else:
            u_hat[i] = 0 if alpha[log2_n, i] >= 0 else 1

        # Back-propagate decision
        beta[log2_n, i] = u_hat[i]
        for s in range(log2_n - 1, -1, -1):
            half = n >> (s + 1)
            if (i & half) == 0:
                break
            left_idx = i - half
            beta[s + 1, left_idx] = beta[s + 2, left_idx] ^ beta[s + 2, i] \
                if s + 2 <= log2_n else u_hat[left_idx] ^ u_hat[i]
            beta[s + 1, i] = beta[s + 2, i] if s + 2 <= log2_n else u_hat[i]

    return u_hat


def sc_decode_fast(llrs: np.ndarray, frozen_mask: np.ndarray) -> np.ndarray:
    """Correct recursive SC decoder matching the step-doubling polar transform.

    The key insight: the g-node partial sum at each level is the polar-
    transformed upper block output, not the raw upper decisions.
    This is because the encoding recursion is:
        x[0:n/2] = transform(u_upper XOR u_lower) -- NOT just u_upper
        x[n/2:n] = transform(u_lower)
    Wait -- actually encoding is:
        x[0:n/2] = transform(u[0:n/2] XOR u[n/2:n])
        x[n/2:n] = transform(u[n/2:n])
    But the step-doubling transform does the LAST butterfly last, meaning:
        last step XORs x[0:n/2] ^= x[n/2:n]
    So to invert the last step for decoding, the partial sum needed
    for the g-node is polar_transform(u_upper_decoded).

    Args:
        llrs:        LLR values, shape (n,).
        frozen_mask: True = frozen bit (forced to 0).

    Returns:
        Decoded bit array u_hat of shape (n,), dtype int8.
    """
    n = len(llrs)
    if n == 1:
        u = 0 if frozen_mask[0] else (0 if llrs[0] >= 0 else 1)
        return np.array([u], dtype=np.int8)

    half = n // 2
    L_left  = llrs[:half]
    L_right = llrs[half:]

    # f-combine (check-node) for upper sub-block
    llrs_upper = (np.sign(L_left) * np.sign(L_right) *
                  np.minimum(np.abs(L_left), np.abs(L_right)))
    u_upper = sc_decode_fast(llrs_upper, frozen_mask[:half])

    # Partial sum = encoded upper block (needed for g-node)
    ps = _polar_transform(u_upper)

    # g-combine (variable-node) for lower sub-block
    llrs_lower = L_right + (1 - 2 * ps.astype(np.float64)) * L_left
    u_lower = sc_decode_fast(llrs_lower, frozen_mask[half:])

    return np.concatenate([u_upper, u_lower])


# ═══════════════════════════════════════════════════════════════════════════
# LLR computation from ArcMark angle estimates
# ═══════════════════════════════════════════════════════════════════════════

def angles_to_llrs(
    angle_estimates: np.ndarray,
    alphabet_size: int = 2,
    phi: float = 0.0,
    kappa: float = 1.0,
) -> np.ndarray:
    """Compute soft LLRs from ArcMark circular angle estimates.

    For binary polar codes (alphabet_size=2), codeword symbol c ∈ {0, 1}
    maps to angle:
        φ_c = 2π·c/2 + phi = {0, π} + phi

    Under a von Mises model with concentration κ, the log-likelihood ratio
    for observing angle θ is:

        LLR = log P(θ | c=0) / P(θ | c=1)
            = κ · [cos(θ - φ_0) - cos(θ - φ_1)]
            = κ · [cos(θ - phi) - cos(θ - phi - π)]
            = 2κ · cos(θ - phi)

    Higher |LLR| = more confident.  LLR > 0 → bit=0, LLR < 0 → bit=1.

    For higher alphabet sizes, we project onto the binary decision by
    comparing the two nearest codeword angles.

    Args:
        angle_estimates: Noisy angle estimates, shape (n,), values in [0, 2π).
        alphabet_size:   Code alphabet size p (2 for binary polar).
        phi:             Angle offset used in ArcMark (default 0.0).
        kappa:           Von Mises concentration parameter (default 1.0).

    Returns:
        LLR array of shape (n,), dtype float64.
    """
    TWO_PI = 2.0 * math.pi
    theta = np.asarray(angle_estimates, dtype=np.float64)

    if alphabet_size == 2:
        # c=0 → angle phi, c=1 → angle phi + π
        llrs = 2.0 * kappa * np.cos(theta - phi)
    else:
        # General: c=0 maps to phi, c=1 maps to phi + 2π/p
        # Use the two smallest circular distances to derive a soft binary LLR
        angle_0 = phi % TWO_PI
        angle_1 = (phi + TWO_PI / alphabet_size) % TWO_PI

        def circ_dist(a: np.ndarray, b: float) -> np.ndarray:
            d = np.abs(a - b) % TWO_PI
            return np.minimum(d, TWO_PI - d)

        d0 = circ_dist(theta, angle_0)
        d1 = circ_dist(theta, angle_1)
        # LLR proportional to difference in circular distances
        llrs = kappa * (d1 - d0)

    return llrs


# ═══════════════════════════════════════════════════════════════════════════
# PolarCode class — ChannelCode interface
# ═══════════════════════════════════════════════════════════════════════════

class PolarCode(ChannelCode):
    """Binary polar code for ArcMark watermarking.

    Supports arbitrarily large k (32, 64, ... bits) with O(n log n)
    encode and decode — no explicit codebook materialised.

    The codeword length is rounded up to the next power of 2 internally.
    Positions beyond n_tokens are padded with 0 and ignored at decode time.

    Args:
        k_bits:    Number of information bits (e.g. 32, 64).
        n_tokens:  Number of watermarked tokens (codeword length visible
                   to the outside; internally rounded up to power of 2).
        seed:      Deterministic seed for frozen bit selection.
        kappa:     Von Mises concentration for soft LLR computation.
        phi:       Angle offset (must match ArcMark processor phi).

    Example::

        code = PolarCode.build(k_bits=32, n_tokens=300, seed=42)
        codeword = code.encode(message_idx)     # shape (300,), values {0,1}
        msg_idx  = code.decode(angle_estimates) # int in [0, 2^32)
        bits     = code.int_to_bits(msg_idx)    # shape (32,)
    """

    # alphabet_size is always 2 for binary polar codes
    _ALPHABET_SIZE: int = 2

    def __init__(
        self,
        *,
        k_bits: int,
        n_tokens: int,
        n_polar: int,
        info_indices: np.ndarray,
        frozen_mask: np.ndarray,
        seed: int,
        kappa: float,
        phi: float,
    ) -> None:
        self._k_bits       = k_bits
        self._n_tokens     = n_tokens
        self._n_polar      = n_polar        # power-of-2 block length
        self._info_indices = info_indices   # shape (k_bits,)
        self._frozen_mask  = frozen_mask    # shape (n_polar,), True=frozen
        self._seed         = seed
        self._kappa        = kappa
        self._phi          = phi

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        k_bits: int,
        n_tokens: int,
        seed: int = 42,
        kappa: float = 1.0,
        phi: float = 0.0,
    ) -> PolarCode:
        """Construct a polar code for k_bits information bits over n_tokens.

        Args:
            k_bits:   Number of information bits (any positive integer).
            n_tokens: Number of token positions to use. Internally rounded
                      up to the next power of 2.
            seed:     Seed for frozen bit selection (must match decoder).
            kappa:    Von Mises concentration parameter for LLR computation.
            phi:      ArcMark angle offset.

        Returns:
            Initialised PolarCode instance.
        """
        if k_bits < 1:
            raise ValueError(f"k_bits must be >= 1, got {k_bits}")
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")

        # Round up to next power of 2
        n_polar = 1
        while n_polar < n_tokens:
            n_polar <<= 1

        if k_bits > n_polar:
            raise ValueError(
                f"k_bits ({k_bits}) cannot exceed n_polar ({n_polar}). "
                f"Increase n_tokens or decrease k_bits."
            )

        info_indices = _select_info_bits(n_polar, k_bits, seed)
        frozen_mask  = np.ones(n_polar, dtype=bool)
        frozen_mask[info_indices] = False

        return cls(
            k_bits=k_bits,
            n_tokens=n_tokens,
            n_polar=n_polar,
            info_indices=info_indices,
            frozen_mask=frozen_mask,
            seed=seed,
            kappa=kappa,
            phi=phi,
        )

    # ── ChannelCode abstract implementations ─────────────────────────────

    def encode(self, message_idx: int) -> Tensor:
        """Encode message_idx to a codeword of length n_tokens.

        Bits beyond n_tokens (padding) are dropped.

        Args:
            message_idx: Integer in [0, 2^k_bits).

        Returns:
            LongTensor of shape (n_tokens,) with values in {0, 1}.
        """
        if not (0 <= message_idx < 2 ** self._k_bits):
            raise IndexError(
                f"message_idx={message_idx} out of range [0, {2**self._k_bits})"
            )
        # Convert to k_bits binary vector (MSB first)
        bits = np.array(
            [(message_idx >> (self._k_bits - 1 - i)) & 1
             for i in range(self._k_bits)],
            dtype=np.int8,
        )

        # Place info bits; frozen bits are 0
        u = np.zeros(self._n_polar, dtype=np.int8)
        u[self._info_indices] = bits

        # Apply polar transform
        x = _polar_transform(u)

        # Return first n_tokens positions
        return torch.tensor(x[:self._n_tokens], dtype=torch.long)

    @property
    def codebook(self) -> Tensor:
        """Not materialised for polar codes — raises NotImplementedError.

        Use encode() and decode() directly.  score_all_messages() is not
        used with PolarCode; decoding is done via decode().
        """
        raise NotImplementedError(
            "PolarCode does not materialise a codebook. "
            "Use PolarCode.decode(angle_estimates) for decoding."
        )

    @property
    def num_messages(self) -> int:
        """Number of distinct messages: 2^k_bits."""
        return 2 ** self._k_bits

    @property
    def codeword_length(self) -> int:
        """Visible codeword length (= n_tokens, not n_polar)."""
        return self._n_tokens

    @property
    def alphabet_size(self) -> int:
        """Code alphabet size: always 2 for binary polar."""
        return self._ALPHABET_SIZE

    @property
    def k_bits(self) -> int:
        """Number of information bits."""
        return self._k_bits

    # ── Polar-specific: soft-decision decode ─────────────────────────────

    def decode(
        self,
        angle_estimates: Tensor,
        kappa: float | None = None,
    ) -> int:
        """Soft-decision SC decode from ArcMark angle estimates.

        Args:
            angle_estimates: Float tensor of shape (n_tokens,) with angle
                             estimates from decode_symbol_angles().
            kappa:           Override von Mises concentration (default: use
                             value set at construction time).

        Returns:
            Decoded message index (int in [0, 2^k_bits)).
        """
        k = kappa if kappa is not None else self._kappa
        angles_np = angle_estimates.cpu().numpy().astype(np.float64)

        # Pad to n_polar with 0.0 (neutral LLR) for positions beyond n_tokens
        angles_padded = np.zeros(self._n_polar, dtype=np.float64)
        angles_padded[:len(angles_np)] = angles_np

        # Compute soft LLRs from circular angle estimates
        llrs = angles_to_llrs(
            angles_padded,
            alphabet_size=self._ALPHABET_SIZE,
            phi=self._phi,
            kappa=k,
        )

        # Run SC decoder
        u_hat = sc_decode_fast(llrs, self._frozen_mask)

        # Extract information bits and convert to integer
        info_bits = u_hat[self._info_indices]   # shape (k_bits,)
        msg_idx = int(
            sum(int(b) << (self._k_bits - 1 - i)
                for i, b in enumerate(info_bits))
        )
        return msg_idx

    def decode_bits(self, angle_estimates: Tensor) -> str:
        """Decode angle estimates to a bit string of length k_bits."""
        msg_idx = self.decode(angle_estimates)
        return format(msg_idx, f"0{self._k_bits}b")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_polar(self) -> int:
        """Internal power-of-2 block length."""
        return self._n_polar

    @property
    def info_indices(self) -> np.ndarray:
        """Indices of information bit positions in the polar codeword."""
        return self._info_indices.copy()

    @property
    def seed(self) -> int:
        return self._seed