"""Hash-based and fixed key generation for ArcMark.

Provides two modes of side-information key generation:

1. **Hash mode** (production): Keys ``(s_index, perm_seed)`` are derived
   by hashing the previous ``context_width`` watermarked tokens together
   with a secret key using SHA-256.  The hash does *not* include the
   token position explicitly — position is implicit in the context token
   sequence.

2. **Fixed mode** (debug / testing): Keys are pre-generated from a
   seeded ``torch.Generator``, independent of token context.  Useful
   for unit tests and deterministic debugging.

Both modes produce ``(s_index, perm_seed)`` pairs.  The decoder
reconstructs keys via :func:`compute_keys_from_tokens` (hash mode)
or :func:`generate_fixed_key_sequence` (fixed mode) using only the
watermarked token sequence and the shared secret — no prompt tokens
are needed.

All operations in this module are **integer-only** (SHA-256,
``struct.pack``, modular arithmetic).  There are no floating-point
operations, so there is zero risk of float32/float64 inconsistencies
between encoder and decoder running on different hardware.

How to read this file
---------------------
Each token position in the watermarked text needs two secret values:

- ``s_index``  (int in {0, …, r-1}):  Selects which side-information
  angle to use.  Combined with the codeword symbol, it determines the
  target angle z_t on the unit circle that the OT solver biases toward.
  In the paper, this is V_t.

- ``perm_seed`` (large int):  Seeds a deterministic random permutation
  of the vocabulary.  The permutation shuffles the token-to-angle
  mapping so an adversary cannot predict it.  In the paper, this
  is Π_t.

Together, (s_index, perm_seed) form the shared secret S_t = (V_t, Π_t).

The **encoder** (``processor.py``) calls ``compute_key()`` once per
token during generation.  The **decoder** calls
``compute_keys_from_tokens()`` after receiving the full watermarked
text, and recovers the same (s_index, perm_seed) at every position.
"""

from __future__ import annotations

import hashlib
import struct

import torch
from torch import Tensor

# ── Side-info module import ───────────────────────────────────────────────
# keygen.py now delegates to side_info.py for all key derivation.
# The original compute_key / compute_keys_from_tokens signatures are
# preserved as thin wrappers so existing code continues to work unchanged.
from arcmark.side_info import (
    SideInfoMode,
    compute_key_si,
    compute_keys_from_tokens_si,
    generate_fixed_key_sequence_si,
)

__all__ = [
    "compute_key",
    "compute_keys_from_tokens",
    "generate_fixed_key_sequence",
    # Also re-export side_info symbols for callers that want the new API
    "SideInfoMode",
    "compute_key_si",
    "compute_keys_from_tokens_si",
]


# ═══════════════════════════════════════════════════════════════════════════
# Hash-based key derivation
# ═══════════════════════════════════════════════════════════════════════════


def _hash_context(
    secret_key: int,
    context_tokens: tuple[int, ...],
) -> bytes:
    """SHA-256 hash of ``(secret_key, context_tokens)``.

    Uses ``struct.pack`` with explicit little-endian int64 encoding for
    each element, producing an unambiguous binary representation that is
    deterministic across platforms and architectures.

    Args:
        secret_key:     Shared secret (packed as signed int64).
        context_tokens: Tuple of preceding token IDs (each packed as
                        signed int64).

    Returns:
        Raw 32-byte SHA-256 digest.
    """
    # struct.pack("<q", x) converts the integer x into exactly 8 bytes
    # using little-endian byte order ("<") and signed 64-bit format ("q").
    # This gives us a fixed-size binary representation — Python ints have
    # variable size, so we need this to get a consistent byte string.
    buf = struct.pack("<q", secret_key)

    # Append each context token as another 8-byte chunk.
    # The final byte string is: [secret_key | tok_0 | tok_1 | ... | tok_{h-1}]
    # where h = context_width.
    for tok in context_tokens:
        buf += struct.pack("<q", tok)

    # Feed the concatenated bytes into SHA-256 and return the raw 32-byte
    # digest (not the hex string — we'll slice raw bytes in compute_key).
    return hashlib.sha256(buf).digest()


def compute_key(
    secret_key: int,
    context_tokens: tuple[int, ...],
    num_keys: int,
) -> tuple[int, int]:
    """Derive ``(s_index, perm_seed)`` for one token position.

    Backward-compatible wrapper around :func:`~arcmark.side_info.compute_key_si`
    using :attr:`~arcmark.side_info.SideInfoMode.HASH_CONTEXT` (original
    behaviour).  Use ``compute_key_si`` directly to select a different mode.
    """
    return compute_key_si(
        secret_key=secret_key,
        context_tokens=context_tokens,
        num_keys=num_keys,
        mode=SideInfoMode.HASH_CONTEXT,
    )


def compute_keys_from_tokens(
    secret_key: int,
    tokens: Tensor,
    context_width: int,
    num_keys: int,
) -> list[tuple[int, int]]:
    """Compute ``(s_index, perm_seed)`` for every position in a token sequence.

    Backward-compatible wrapper around
    :func:`~arcmark.side_info.compute_keys_from_tokens_si` using
    :attr:`~arcmark.side_info.SideInfoMode.HASH_CONTEXT`.
    """
    return compute_keys_from_tokens_si(
        secret_key=secret_key,
        tokens=tokens,
        context_width=context_width,
        num_keys=num_keys,
        mode=SideInfoMode.HASH_CONTEXT,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fixed (context-independent) key generation
# ═══════════════════════════════════════════════════════════════════════════


def generate_fixed_key_sequence(
    seed: int,
    length: int,
    num_keys: int,
) -> list[tuple[int, int]]:
    """Generate a context-independent key sequence.

    Backward-compatible wrapper around
    :func:`~arcmark.side_info.generate_fixed_key_sequence_si`.
    """
    return generate_fixed_key_sequence_si(seed=seed, length=length, num_keys=num_keys)