"""Variable-length Laplace-channel watermark decoder for ArcMark.

This module implements an adaptive decoder that accumulates information
density token by token and stops early when confident enough — instead
of always decoding at a fixed checkpoint.

**Channel model:**

Assumes the angle residual at position t follows a Laplace distribution:

    θ_t | m  ~  Laplace(μ_{m,t}, b)

where μ_{m,t} = 2π · codeword[m,t] / p is the expected angle for message
m at position t, and b is the scale hyperparameter.

**Information density accumulator:**

For each candidate message m:

    I_m(t) = Σ_{s=1}^{t} [ log f(θ_s | m) - log f_N(θ_s) ]

where:
    log f(θ_s | m) = -|θ_s - μ_{m,s}|_circ / b      (Laplace log-likelihood)
    log f_N(θ_s)   = logsumexp_m(-|θ_s - μ_{m,s}|_circ / b) - log(M)
                     (mixture marginal — uniform prior over messages)

For the **gap** stopping rule, log f_N cancels (same for all m) and is
skipped for efficiency.

**Stopping rules:**

    gap:       stop at t when I_best(t) - I_second(t) >= η
    best_info: stop at t when I_best(t) >= τ

**Key design:** paths are computed ONCE over all tokens, then any number
of thresholds can be applied offline with no re-decoding.

Usage::

    from vl_decoder import compute_info_density_paths, apply_threshold

    paths = compute_info_density_paths(
        tokens=arc_tokens,
        code=code,
        config=arcmark_config,
        vocab_size=vocab_size,
        alphabet_size=alphabet_size,
        num_keys=num_keys,
        secret_key=ex_secret_key,
        laplace_b=0.5,
        side_info_mode=side_info_mode,
        tokenizer=tokenizer,
    )

    # Apply multiple thresholds offline — no re-decoding
    for eta in [3.0, 5.0, 10.0]:
        result = apply_threshold(paths, "gap", eta, t_min=10)
        print(result["t_stop"], result["decoded_bits"])
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from arcmark.config import ArcMarkConfig
from arcmark.side_info import SideInfoMode
from arcmark.symbol_decoder import decode_symbol_angles

__all__ = [
    "compute_info_density_paths",
    "apply_threshold",
]

TWO_PI = 2.0 * math.pi


def compute_info_density_paths(
    tokens: list[int],
    code,
    config: ArcMarkConfig,
    vocab_size: int,
    alphabet_size: int,
    num_keys: int,
    secret_key: int,
    laplace_b: float = 0.5,
    side_info_mode: SideInfoMode = SideInfoMode.HASH_CONTEXT,
    tokenizer: Any = None,
) -> dict:
    """Compute per-token information density paths for all messages.

    Runs through all tokens ONCE and accumulates I_m for every message m.
    Returns paths so that any number of thresholds can be evaluated
    offline with no re-decoding.

    Args:
        tokens:          Full token list (up to MAX_TOKENS).
        code:            RandomLinearCode or GFCode instance.
        config:          ArcMarkConfig (context_width, hash_keys, etc.).
        vocab_size:      Model vocabulary size.
        alphabet_size:   Code alphabet size p.
        num_keys:        Side-information cardinality r.
        secret_key:      Per-example secret key.
        laplace_b:       Laplace scale parameter b. Smaller = sharper
                         likelihoods, faster stopping, higher error risk.
        side_info_mode:  Key derivation mode (must match encoder).
        tokenizer:       HuggingFace tokenizer (required for normalized
                         and char_ngram side-info modes).

    Returns:
        dict with:
            best_I_path:    (n,) float64 — best I_m at each token position
            second_I_path:  (n,) float64 — second-best I_m at each position
            gap_path:       (n,) float64 — best minus second at each position
            leader_path:    (n,) int32   — argmax message index at each position
            all_I:          (M,) float64 — final accumulated I_m for all messages
            n:              int           — number of tokens processed
    """
    M      = code.num_messages
    p      = alphabet_size
    n_full = len(tokens)

    # Precompute codeword angles: shape (M, n_full)
    cb              = code.codebook.numpy().astype(np.float64)   # (M, n_full)
    codeword_angles = TWO_PI * cb / float(p)                     # (M, n_full)

    # Decode all angle estimates at once — one forward pass through keys
    toks_tensor = torch.tensor(tokens, dtype=torch.long)
    all_angles  = decode_symbol_angles(
        toks_tensor,
        vocab_size=vocab_size,
        num_keys=num_keys,
        seed=secret_key,
        config=config,
        side_info_mode=side_info_mode,
        tokenizer=tokenizer,
    ).numpy()   # (n_full,) float64

    # Accumulators
    all_I         = np.zeros(M, dtype=np.float64)
    best_I_path   = np.zeros(n_full, dtype=np.float64)
    second_I_path = np.zeros(n_full, dtype=np.float64)
    gap_path      = np.zeros(n_full, dtype=np.float64)
    leader_path   = np.zeros(n_full, dtype=np.int32)

    for t in range(n_full):
        theta = all_angles[t]                       # scalar
        mu_t  = codeword_angles[:, t]               # (M,)

        # Circular distance in [0, π]
        diff = np.abs(theta - mu_t)
        circ = np.minimum(diff, TWO_PI - diff)      # (M,)

        # Laplace log-likelihoods (constant -log(2b) dropped, same for all m)
        log_liks = -circ / laplace_b                # (M,)

        # Null: log f_N(θ_t) = logsumexp(log_liks) - log(M)
        # This is the same for all m so it shifts all I_m equally.
        # For the gap rule it cancels; for best_info it normalises the scale.
        log_fN = float(np.logaddexp.reduce(log_liks)) - math.log(M)

        all_I += log_liks - log_fN

        # Track top-2 for gap path
        if M >= 2:
            # argpartition is O(M) — fast
            top2_idx = np.argpartition(-all_I, 2)[:2]
            if all_I[top2_idx[0]] >= all_I[top2_idx[1]]:
                best_idx, second_idx = top2_idx[0], top2_idx[1]
            else:
                best_idx, second_idx = top2_idx[1], top2_idx[0]
        else:
            best_idx = second_idx = 0

        best_I_path[t]   = all_I[best_idx]
        second_I_path[t] = all_I[second_idx]
        gap_path[t]      = all_I[best_idx] - all_I[second_idx]
        leader_path[t]   = best_idx

    return {
        "best_I_path":   best_I_path,
        "second_I_path": second_I_path,
        "gap_path":      gap_path,
        "leader_path":   leader_path,
        "all_I":         all_I,
        "n":             n_full,
    }


def apply_threshold(
    paths: dict,
    stop_rule: str,
    threshold: float,
    t_min: int = 10,
) -> dict:
    """Apply a stopping threshold to precomputed information density paths.

    This is a pure offline operation — no model or token access needed.
    Call it once per threshold after a single compute_info_density_paths call.

    Args:
        paths:      Output dict from compute_info_density_paths.
        stop_rule:  "gap"       — stop when best_I - second_I >= threshold.
                    "best_info" — stop when best_I >= threshold.
        threshold:  Stopping threshold value.
        t_min:      Minimum number of tokens before stopping is allowed.
                    1-indexed (t_min=10 means at least 10 tokens are used).

    Returns:
        dict with:
            best_m:       int   — decoded message index
            t_stop:       int   — number of tokens used (1-indexed)
            gap_final:    float — gap at stopping time
            best_I_final: float — best I_m at stopping time
    """
    best_I_path = paths["best_I_path"]
    gap_path    = paths["gap_path"]
    leader_path = paths["leader_path"]
    n_full      = paths["n"]

    t0   = max(1, int(t_min))
    path = gap_path if stop_rule == "gap" else best_I_path

    # Find first position >= t_min where threshold is exceeded
    hits = np.where(path[t0 - 1:] >= threshold)[0]

    if hits.size > 0:
        t_stop = int(t0 - 1 + hits[0] + 1)   # convert to 1-indexed
    else:
        t_stop = n_full                        # use all tokens

    best_m = int(leader_path[t_stop - 1])

    return {
        "best_m":       best_m,
        "t_stop":       t_stop,
        "gap_final":    float(gap_path[t_stop - 1]),
        "best_I_final": float(best_I_path[t_stop - 1]),
    }
