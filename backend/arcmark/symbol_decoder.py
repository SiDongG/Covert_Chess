"""Symbol-level decoder for ArcMark watermark detection.

Maps each watermarked token + side-information key to an estimated
codeword symbol angle on the unit circle.  This is the first stage
of the ArcMark decoder pipeline:

1. **Symbol decoding** (this module): token + key -> angle estimate
2. **Message decoding** (``message_decoder.py``): angle sequence -> message

For each token position *t*, the symbol decoder:

1. Reconstructs the key pair ``(s_index, perm_seed)`` via hash or fixed mode.
2. Rebuilds the permutation Pi_t from ``perm_seed``.
3. Computes the permuted token angle: theta_t = 2*pi * Pi_t(x_t) / N.
4. Removes the shared randomness:
   C_hat(t) = (theta_t - 2*pi * s_index / r) mod 2*pi.

The output C_hat(t) is a noisy estimate of the true codeword symbol
angle ``2*pi * C_m(t) / p + phi``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from arcmark import geometry
from arcmark.config import ArcMarkConfig
from arcmark.keygen import compute_keys_from_tokens, generate_fixed_key_sequence
from arcmark.side_info import SideInfoMode, compute_keys_from_tokens_si

__all__ = [
    "decode_symbol_angle",
    "decode_symbol_angles",
]


# ═══════════════════════════════════════════════════════════════════════════
# Single-position decoder
# ═══════════════════════════════════════════════════════════════════════════


def decode_symbol_angle(
    token_id: int,
    s_index: int,
    perm_seed: int,
    vocab_size: int,
    num_keys: int,
) -> float:
    """Compute the estimated codeword symbol angle for one token position.

    Inverts the encoder's angle computation:

    1. Rebuild permutation: Pi = random_permutation(vocab_size, perm_seed)
    2. Permuted token angle: theta = 2*pi * Pi(token_id) / N
    3. Remove shared randomness: c_hat = (theta - 2*pi * s_index / r) mod 2*pi

    The output approximates ``2*pi * C_m(t) / p + phi``, where C_m(t) is
    the true codeword symbol and phi is the angle offset.

    Args:
        token_id:   Observed token ID (int in {0, ..., vocab_size-1}).
        s_index:    Side-information key V_t (int in {0, ..., num_keys-1}).
        perm_seed:  Seed for the deterministic permutation Pi_t.
        vocab_size: Total vocabulary size N.
        num_keys:   Side-information alphabet size r.

    Returns:
        Estimated codeword symbol angle in [0, 2*pi).
    """
    perm = geometry.random_permutation(vocab_size, seed=perm_seed)
    permuted_id = int(perm[token_id].item())

    # Compute in float64 to avoid precision loss for large vocab sizes
    theta = geometry.TWO_PI * permuted_id / float(vocab_size)
    s_angle = geometry.TWO_PI * s_index / float(num_keys)
    c_hat = (theta - s_angle) % geometry.TWO_PI
    return float(c_hat)


# ═══════════════════════════════════════════════════════════════════════════
# Batch decoder (full sequence)
# ═══════════════════════════════════════════════════════════════════════════


def decode_symbol_angles(
    tokens: Tensor,
    *,
    vocab_size: int,
    num_keys: int,
    seed: int,
    config: ArcMarkConfig | None = None,
    key_sequence: list[tuple[int, int]] | None = None,
    side_info_mode: SideInfoMode = SideInfoMode.HASH_CONTEXT,
    tokenizer: Any = None,
) -> Tensor:
    """Compute estimated codeword symbol angles for an entire token sequence.

    Args:
        tokens:         1-D LongTensor of watermarked token IDs, shape (n,).
        vocab_size:     Total vocabulary size N.
        num_keys:       Side-information alphabet size r.
        seed:           Shared secret seed (same as encoder).
        config:         ArcMarkConfig controlling key generation mode.
        key_sequence:   Optional explicit key pairs (overrides key generation).
        side_info_mode: Side-information strategy — must match the encoder.
        tokenizer:      HuggingFace tokenizer (required for NORMALIZED /
                        CHAR_NGRAM modes).

    Returns:
        Float64 tensor of shape (n,) with estimated codeword symbol angles.
    """
    cfg = config if config is not None else ArcMarkConfig()
    n   = len(tokens)

    if n == 0:
        return torch.empty(0, dtype=torch.float64)

    if key_sequence is not None:
        if len(key_sequence) != n:
            raise ValueError(
                f"key_sequence length ({len(key_sequence)}) does not match "
                f"tokens length ({n})"
            )
        keys = key_sequence
    elif cfg.hash_keys:
        keys = compute_keys_from_tokens_si(
            secret_key=seed,
            tokens=tokens,
            context_width=cfg.context_width,
            num_keys=num_keys,
            mode=side_info_mode,
            tokenizer=tokenizer,
        )
    else:
        keys = generate_fixed_key_sequence(
            seed=seed, length=n, num_keys=num_keys,
        )

    angles = torch.empty(n, dtype=torch.float64)
    for t in range(n):
        s_index, perm_seed = keys[t]
        angles[t] = decode_symbol_angle(
            token_id=int(tokens[t].item()),
            s_index=s_index,
            perm_seed=perm_seed,
            vocab_size=vocab_size,
            num_keys=num_keys,
        )

    return angles