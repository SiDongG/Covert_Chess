"""Hash-column code for ArcMark watermarking.

This module provides :class:`HashColumnCode` — a replacement for
:class:`~arcmark.coding.RandomLinearCode` where every column of the
generator matrix is derived on-the-fly from a hash of the context tokens,
rather than being fixed at build time.

**Construction:**

For each token position t:

    g_t = hash_to_vector(secret_key, context_tokens_t)  ∈ Z_p^d
    codeword[m, t] = m_vec(m) · g_t  mod p

where m_vec(m) is the base-p representation of message index m
(same natural enumeration as EfficientRandomLinearCode).

**Key properties:**

1. No fixed generator matrix stored — G is ephemeral and position-dependent
2. The codebook is secret (depends on secret_key) and context-dependent
3. After a back-translation attack, drift in g_t is limited to exactly
   context_width positions per changed token — same as side-info drift
4. The side-info shift s_index is orthogonal and unchanged
5. Compatible with the standard minimum-distance decoder via
   score_all_messages — but requires the context-aware codebook to be
   passed in rather than the fixed one

**Usage in evaluation:**

At decode time, the decoder reconstructs g_t from the observed tokens
(same hash as encoder) and builds the effective codebook columns on the fly.
This is handled by HashColumnDecoder.decode().
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "HashColumnCode",
    "HashColumnDecoder",
    "hash_to_vector",
    "build_context_codebook",
    "build_context_Gv",
    "decode_hash_column_efficient",
]


# ─────────────────────────────────────────────────────────────────────────────
# Hash helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_to_vector(
    secret_key: int,
    context_tokens: tuple[int, ...],
    d: int,
    p: int,
) -> np.ndarray:
    """Derive a d-dimensional vector in Z_p^d from a hash.

    Uses SHA-256(secret_key || context_tokens) to produce a deterministic
    vector g_t ∈ {0,...,p-1}^d.  The same inputs always produce the same
    output — both encoder and decoder call this with identical arguments.

    Args:
        secret_key:      Shared integer secret (same as ArcMark seed).
        context_tokens:  Tuple of preceding watermarked token IDs.
                         Length = context_width (padded with 0s at start).
        d:               Message vector dimension = ceil(k / log2(p)).
        p:               Alphabet size.

    Returns:
        int32 array of shape (d,) with values in {0,...,p-1}.
    """
    buf = struct.pack("<q", secret_key)
    for tok in context_tokens:
        buf += struct.pack("<q", tok)
    digest = hashlib.sha256(buf).digest()

    # Extract d values from the digest, each in {0,...,p-1}
    # Use successive 4-byte chunks, cycling through digest if needed
    g = np.zeros(d, dtype=np.int32)
    for j in range(d):
        byte_offset = (j * 4) % (len(digest) - 3)
        val = int.from_bytes(digest[byte_offset: byte_offset + 4], "little")
        g[j] = val % p
    return g


def build_context_codebook(
    secret_key: int,
    context_width: int,
    tokens: list[int],
    n: int,
    k_bits: int,
    p: int,
) -> Tensor:
    """Build the full context-dependent codebook for n positions.

    Computes codeword[m, t] = m_vec(m) · g_t mod p for all messages m
    and positions t = 0..n-1, where g_t is derived from the hash of
    the context tokens at position t.

    This materializes the full (M x n) codebook in memory — only feasible
    for small k (up to 16 bits / M=65536).  For larger k use
    HashColumnDecoder.score_message() which avoids materializing the codebook.

    Args:
        secret_key:      Shared integer secret.
        context_width:   Number of preceding tokens in context.
        tokens:          The watermarked token IDs (without prompt).
                         Length must be >= n.
        n:               Number of positions to build codebook for.
        k_bits:          Number of information bits.
        p:               Alphabet size.

    Returns:
        LongTensor of shape (M, n) where M = 2^k_bits.
        Values in {0,...,p-1}.
    """
    M = 1 << k_bits
    d = max(1, int(math.ceil(k_bits * math.log(2) / math.log(p))))

    # Precompute all message vectors: m_vec[m] = base-p digits of m
    m_indices = np.arange(M, dtype=np.int64)
    m_vecs = np.zeros((M, d), dtype=np.int64)
    for j in range(d):
        m_vecs[:, j] = (m_indices // (p ** j)) % p

    codebook = np.zeros((M, n), dtype=np.int64)

    for t in range(n):
        # Build context tuple (pad with 0s at start)
        start = max(0, t - context_width)
        ctx   = tokens[start:t]
        pad   = context_width - len(ctx)
        context = tuple([0] * pad + list(ctx))

        # Hash-derived column vector
        g_t = hash_to_vector(secret_key, context, d, p).astype(np.int64)

        # codeword[m, t] = m_vec[m] · g_t  mod p  for all m at once
        codebook[:, t] = (m_vecs @ g_t) % p

    return torch.tensor(codebook, dtype=torch.long)


def build_context_Gv(
    secret_key: int,
    context_width: int,
    tokens: list[int],
    n: int,
    d: int,
    p: int,
) -> np.ndarray:
    """Build Gv[j, v, t] = v * g_t[j] mod p for all j, v, t.

    This is the hash-column analogue of EfficientRandomLinearCode._Gv.
    Instead of a fixed G, each column g_t is derived from the context hash.
    Used by the efficient nested decoder for k > 16.

    Args:
        secret_key:     Shared integer secret.
        context_width:  Number of preceding tokens in context.
        tokens:         Observed token IDs (without prompt), length >= n.
        n:              Number of positions.
        d:              Message vector dimension = ceil(k / log2(p)).
        p:              Alphabet size.

    Returns:
        int32 array of shape (d, p, n).
        Gv[j, v, t] = (v * g_t[j]) mod p.
    """
    v_vals = np.arange(p, dtype=np.int64)   # (p,)
    Gv     = np.zeros((d, p, n), dtype=np.int32)

    for t in range(n):
        start   = max(0, t - context_width)
        ctx     = tokens[start:t]
        pad     = context_width - len(ctx)
        context = tuple([0] * pad + list(ctx))
        g_t     = hash_to_vector(secret_key, context, d, p).astype(np.int64)  # (d,)

        for j in range(d):
            Gv[j, :, t] = (v_vals * g_t[j]) % p

    return Gv


def _decode_hash_gpu(
    b_np: np.ndarray,
    Gv_np: np.ndarray,
    n: int,
    p: int,
    d: int,
    device: torch.device,
) -> np.ndarray:
    """Nested GPU decoder for hash-column code — identical logic to
    EfficientRandomLinearCode._decode_gpu but with hash-derived Gv.

    Args:
        b_np:   (n, p) float32 beliefs.
        Gv_np:  (d, p, n) int32 — Gv[j,v,t] = v*g_t[j] mod p.
        n, p, d: dimensions.
        device: CUDA device.

    Returns:
        Decoded message vector (d,) int32.
    """
    b    = torch.from_numpy(b_np).to(device)
    Gv   = torch.from_numpy(Gv_np.astype(np.int64)).to(device)
    tidx = torch.arange(n, dtype=torch.int64, device=device)
    v    = torch.arange(p, dtype=torch.int64, device=device)

    def smod(x):
        return x.remainder(p).clamp(0, p - 1)

    # Precompute A01[t, m0, m1] = (Gv[0,m0,t] + Gv[1,m1,t]) mod p
    A01 = smod(Gv[0][:, None, :] + Gv[1][None, :, :])   # (p, p, n)
    A01 = A01.permute(2, 0, 1).contiguous()              # (n, p, p)

    if d == 1:
        sym    = smod(Gv[0])                             # (p, n)
        scores = b[tidx[None, :], sym].sum(dim=1)        # (p,)
        return np.array([int(scores.argmin().item())], dtype=np.int32)

    if d == 2:
        score2 = b[tidx[:, None, None], A01].sum(dim=0)  # (p, p)
        flat   = int(score2.view(-1).argmin().item())
        return np.array([flat // p, flat % p], dtype=np.int32)

    if d == 3:
        best_score, best_flat, best_m2 = float("inf"), 0, 0
        for m2 in range(p):
            base2     = Gv[2, m2]
            shift_idx = smod(v[None, :] + base2[:, None])
            b_shifted = b[tidx[:, None], shift_idx]
            score2    = b_shifted[tidx[:, None, None], A01].sum(dim=0)
            min_val, flat = score2.view(-1).min(dim=0)
            if min_val.item() < best_score:
                best_score, best_flat, best_m2 = min_val.item(), int(flat.item()), m2
        m0, m1 = best_flat // p, best_flat % p
        return np.array([m0, m1, best_m2], dtype=np.int32)

    if d == 4:
        best_score, best_flat, best_m2, best_m3 = float("inf"), 0, 0, 0
        for m3 in range(p):
            base3 = Gv[3, m3]
            for m2 in range(p):
                base23    = smod(base3 + Gv[2, m2])
                shift_idx = smod(v[None, :] + base23[:, None])
                b_shifted = b[tidx[:, None], shift_idx]
                score2    = b_shifted[tidx[:, None, None], A01].sum(dim=0)
                min_val, flat = score2.view(-1).min(dim=0)
                if min_val.item() < best_score:
                    best_score  = min_val.item()
                    best_flat   = int(flat.item())
                    best_m2, best_m3 = m2, m3
        m0, m1 = best_flat // p, best_flat % p
        return np.array([m0, m1, best_m2, best_m3], dtype=np.int32)

    if d == 5:
        best_score, best_flat = float("inf"), 0
        best_m2, best_m3, best_m4 = 0, 0, 0
        for m4 in range(p):
            base4 = Gv[4, m4]
            for m3 in range(p):
                base34 = smod(base4 + Gv[3, m3])
                for m2 in range(p):
                    base234   = smod(base34 + Gv[2, m2])
                    shift_idx = smod(v[None, :] + base234[:, None])
                    b_shifted = b[tidx[:, None], shift_idx]
                    score2    = b_shifted[tidx[:, None, None], A01].sum(dim=0)
                    min_val, flat = score2.view(-1).min(dim=0)
                    if min_val.item() < best_score:
                        best_score  = min_val.item()
                        best_flat   = int(flat.item())
                        best_m2, best_m3, best_m4 = m2, m3, m4
        m0, m1 = best_flat // p, best_flat % p
        return np.array([m0, m1, best_m2, best_m3, best_m4], dtype=np.int32)

    raise NotImplementedError(f"GPU decoder not implemented for d={d} (max d=5).")


def _compute_beliefs(
    angle_estimates: np.ndarray,
    p: int,
    scoring: str = "log",
) -> np.ndarray:
    """Per-position soft beliefs b[t, s], shape (n, p), float32.
    Lower = more likely symbol s at position t."""
    import math as _math
    s_vals   = np.arange(p, dtype=np.float64)
    angles_s = (2.0 * _math.pi * s_vals / p) % (2.0 * _math.pi)
    theta    = angle_estimates.astype(np.float64)
    diffs    = np.abs(theta[:, None] - angles_s[None, :])
    diffs    = np.minimum(diffs, 2.0 * _math.pi - diffs)
    if scoring == "log":
        b = -np.log(np.clip(1.0 - diffs / _math.pi, 1e-12, None))
    else:
        b = diffs
    return b.astype(np.float32)


def decode_hash_column_efficient(
    angles_np: np.ndarray,
    Gv_np: np.ndarray,
    p: int,
    d: int,
    k_bits: int,
    device: torch.device | None = None,
) -> int:
    """Efficient nested decoder for hash-column code.

    Uses the same nested GPU search as EfficientRandomLinearCode but
    with hash-derived Gv instead of a fixed generator matrix.

    Args:
        angles_np:  (n,) float64 — angle estimates after side-info removal.
        Gv_np:      (d, p, n) int32 — from build_context_Gv().
        p:          Alphabet size.
        d:          Message vector dimension.
        k_bits:     Number of information bits.
        device:     CUDA device (auto-detected if None).

    Returns:
        Decoded message index as int.
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else None

    n_actual = len(angles_np)
    Gv_sliced = Gv_np[:, :, :n_actual]   # slice to actual checkpoint length
    b = _compute_beliefs(angles_np, p)

    use_gpu = (device is not None and device.type == "cuda" and d <= 5)

    if use_gpu:
        m_vec = _decode_hash_gpu(b, Gv_sliced, n_actual, p, d, device)
    else:
        raise RuntimeError(
            f"hash_column efficient decoder requires GPU (d={d}, k={k_bits}). "
            f"No CUDA device available."
        )

    # Convert m_vec back to message index
    idx, p_pow = 0, 1
    for j in range(d):
        idx += int(m_vec[j]) * p_pow
        p_pow *= p
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# HashColumnCode — ChannelCode-compatible interface for the encoder
# ─────────────────────────────────────────────────────────────────────────────

class HashColumnCode:
    """Context-dependent hash-column code for ArcMark watermarking.

    The encoder calls encode(message_idx, context_at_t) at each position t
    to get the codeword symbol.  The decoder reconstructs the same symbol
    from the observed tokens using the same hash.

    This class is NOT a ChannelCode subclass because the codebook is not
    fixed — it depends on the actual generated tokens.  Instead it provides:

    - encode_symbol(message_idx, context_tokens):  one symbol at a time
    - build_codebook(secret_key, tokens, n):        full codebook for decoding

    Args:
        k_bits:    Number of information bits.
        p:         Alphabet size.
        secret_key: Shared integer secret.
        context_width: Number of preceding tokens in context.
    """

    def __init__(
        self,
        k_bits: int,
        p: int,
        secret_key: int,
        context_width: int,
    ) -> None:
        self._k     = k_bits
        self._p     = p
        self._d     = max(1, int(math.ceil(k_bits * math.log(2) / math.log(p))))
        self._seed  = secret_key
        self._cw    = context_width
        self._M     = 1 << k_bits

    def encode_symbol(
        self,
        message_idx: int,
        context_tokens: tuple[int, ...],
    ) -> int:
        """Encode one codeword symbol at a position.

        Args:
            message_idx:     Message index in [0, 2^k).
            context_tokens:  Tuple of preceding context_width token IDs
                             (padded with 0s at the start if needed).

        Returns:
            Codeword symbol in {0,...,p-1}.
        """
        g_t   = hash_to_vector(self._seed, context_tokens, self._d, self._p)
        m_vec = self._idx_to_mvec(message_idx)
        return int(np.dot(m_vec.astype(np.int64),
                          g_t.astype(np.int64)) % self._p)

    def _idx_to_mvec(self, idx: int) -> np.ndarray:
        """Convert message index to base-p vector."""
        m = np.empty(self._d, dtype=np.int32)
        val = int(idx)
        for j in range(self._d):
            m[j] = val % self._p
            val //= self._p
        return m

    def int_to_bits(self, message_idx: int) -> Tensor:
        """Convert message index to binary bit vector (MSB first)."""
        bits = []
        val = message_idx
        for _ in range(self._k):
            bits.append(val & 1)
            val >>= 1
        bits.reverse()
        return torch.tensor(bits, dtype=torch.uint8)

    def build_codebook(self, tokens: list[int], n: int) -> Tensor:
        """Build full (M, n) codebook for scoring at decode time."""
        return build_context_codebook(
            secret_key=self._seed,
            context_width=self._cw,
            tokens=tokens,
            n=n,
            k_bits=self._k,
            p=self._p,
        )

    @property
    def k_bits(self) -> int:
        return self._k

    @property
    def num_messages(self) -> int:
        return self._M

    @property
    def alphabet_size(self) -> int:
        return self._p

    @property
    def context_width(self) -> int:
        return self._cw


# ─────────────────────────────────────────────────────────────────────────────
# HashColumnDecoder — decode using context-aware codebook
# ─────────────────────────────────────────────────────────────────────────────

class HashColumnDecoder:
    """Decoder for HashColumnCode watermarks.

    Reconstructs the context-dependent codebook from the observed tokens
    and scores all messages using the standard minimum-distance criterion.

    For k <= 16 (M <= 65536): materializes the full codebook via
    build_context_codebook() and calls score_all_messages().

    Args:
        code:       HashColumnCode instance (same params as encoder).
        vocab_size: Model vocabulary size.
        num_keys:   Side-information cardinality r.
    """

    def __init__(
        self,
        code: HashColumnCode,
        vocab_size: int,
        num_keys: int,
    ) -> None:
        self._code       = code
        self._vocab_size = vocab_size
        self._num_keys   = num_keys

    def decode(
        self,
        tokens: list[int],
        n: int,
        config: Any,
        secret_key: int,
        side_info_mode: Any,
        tokenizer: Any = None,
    ) -> str:
        """Decode watermark using efficient nested GPU decoder.

        Builds Gv from hash-derived columns on the fly, then uses
        EfficientRandomLinearCode's _compute_beliefs + _decode_gpu/_decode_cpu
        — the same proven implementation used for fixed codebooks.

        Args:
            tokens:          Observed token IDs (possibly attacked).
            n:               Number of tokens to decode (checkpoint).
            config:          ArcMarkConfig (for key generation).
            secret_key:      Shared integer secret.
            side_info_mode:  Must match encoder.
            tokenizer:       HuggingFace tokenizer if needed.

        Returns:
            Decoded message as a bit string of length k_bits.
        """
        from arcmark.symbol_decoder import decode_symbol_angles
        from arcmark.efficient_random_linear_code import (
            _compute_beliefs, _decode_gpu, _decode_cpu, _mvec_to_idx
        )

        k_bits = self._code.k_bits
        p      = self._code.alphabet_size
        d      = self._code._d

        # Decode symbol angles (side-info removed)
        toks   = torch.tensor(tokens[:n], dtype=torch.long)
        angles = decode_symbol_angles(
            toks,
            vocab_size=self._vocab_size,
            num_keys=self._num_keys,
            seed=secret_key,
            config=config,
            side_info_mode=side_info_mode,
            tokenizer=tokenizer,
        )  # (n,) float64

        # Build hash-derived Gv for exactly n tokens (checkpoint length)
        Gv_np = build_context_Gv(
            secret_key=secret_key,
            context_width=self._code.context_width,
            tokens=tokens,
            n=n,
            d=d,
            p=p,
        )  # (d, p, n) — already sliced to n, no further slicing needed

        # Compute soft beliefs
        angles_np = angles.numpy().astype(np.float64)
        b = _compute_beliefs(angles_np, p)

        # Nested decoder — same implementation as EfficientRandomLinearCode
        device  = torch.device("cuda") if torch.cuda.is_available() else None
        use_gpu = (device is not None and d <= 5)

        if use_gpu:
            m_vec = _decode_gpu(b, Gv_np, n, p, d, device)
        elif d <= 3:
            m_vec = _decode_cpu(b, Gv_np, n, p, d)
        else:
            raise RuntimeError(
                f"d={d} (k={k_bits}) requires GPU. No CUDA available."
            )

        message_idx = _mvec_to_idx(m_vec, p)
        return "".join(
            str(b) for b in self._code.int_to_bits(message_idx).tolist()
        )
