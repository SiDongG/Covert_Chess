"""Message-level decoder for ArcMark watermark detection.

Takes a sequence of estimated codeword symbol angles (from
:func:`~arcmark.symbol_decoder.decode_symbol_angles`) and decodes the
embedded multi-bit message by scoring all candidate codewords.

**Decoding algorithm:**

For each candidate message m in {0, ..., M-1}:

1. Retrieve its codeword C_m = [C_m(0), ..., C_m(n-1)] from the linear code.
2. Compute the angular codeword representation:
   C_m_ang(t) = 2*pi * C_m(t) / p + phi
3. Compute the total scoring distance:
   D_m = sum_t f( d( C_hat(t), C_m_ang(t) ) )
   where d is circular distance and f is a scoring function.
4. Decode: m_hat = argmin_m D_m

**Scoring functions** (parameter ``scoring``):

- ``"linear"``:  f(d) = d  (simple circular distance)
- ``"log"``:     f(d) = -log(1 - d / d_max)  (capacity-achieving)

**Convenience functions:**

- :func:`score_all_messages` — score all candidates given angles and codewords.
- :func:`decode_message` — full pipeline: tokens -> symbol angles -> message.
- :func:`decode_with_code` — same, but accepts a :class:`~arcmark.coding.ChannelCode`
  object and fills ``message_bits`` in the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from arcmark import geometry

if TYPE_CHECKING:
    from arcmark.coding import ChannelCode
    from arcmark.config import ArcMarkConfig

NOWM_IDX = -1
"""Sentinel message index returned when no watermark is detected."""

# Default per-token score thresholds above which text is declared unwatermarked.
# For random (unwatermarked) text the expected per-token circular distance to
# any codeword angle is ≈ π/2 ≈ 1.571.  The thresholds below sit well below
# that expectation so that only genuinely unwatermarked text triggers them.
_DEFAULT_NOWM_THRESHOLD = {
    "linear": math.pi / 3,   # ≈ 1.047 per token
    "log":    0.6,            # expected random ≈ 1.0 per token
}

__all__ = [
    "ArcMarkDecodeResult",
    "NOWM_IDX",
    "decode_message",
    "decode_with_code",
    "score_all_messages",
]


@dataclass
class ArcMarkDecodeResult:
    """Result container for message-level decoding.

    Attributes:
        message_idx:   Decoded message index (argmin of scores).
        scores:        Score for each candidate message, shape ``(M,)``.
                       Lower is better.
        best_score:    Score of the decoded message.
        message_bits:  Decoded message as a bit tensor (if available).
                       Filled by :func:`decode_with_code`; ``None`` when
                       using the lower-level :func:`decode_message`.
    """

    message_idx: int
    scores: Tensor
    best_score: float
    message_bits: Tensor | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════


def score_all_messages(
    angle_estimates: Tensor,
    codewords: Tensor,
    *,
    alphabet_size: int,
    phi: float = 0.0,
    scoring: Literal["linear", "log"] = "log",
    d_max: float | None = None,
) -> Tensor:
    """Score all candidate messages given estimated codeword symbol angles.

    For each candidate message m, computes:

        D_m = sum_t f( d( angle_estimates[t],  2*pi * codewords[m, t] / p + phi ) )

    where d(.,.) is the circular distance on [0, 2*pi) and f is the
    scoring function.

    Args:
        angle_estimates: Estimated angles from symbol decoder, shape ``(n,)``,
                         dtype float64.
        codewords:       All candidate codewords, shape ``(M, n)``,
                         entries in ``{0, ..., alphabet_size - 1}``.
        alphabet_size:   Code alphabet size p.
        phi:             Angle offset (default 0.0).
        scoring:         ``"linear"`` for f(d) = d,
                         ``"log"`` for f(d) = -log(1 - d / d_max).
        d_max:           Maximum distance for log scoring.  Defaults to pi
                         when ``None``.

    Returns:
        Score tensor of shape ``(M,)``, dtype float64.  Lower is better.
    """
    M, n = codewords.shape

    # Handle n=0 edge case: all scores are 0
    if n == 0:
        return torch.zeros(M, dtype=torch.float64)

    # 1. Compute angular codeword representations:
    #    C_m_ang(t) = (2*pi * C_m(t) / p + phi) mod 2*pi
    #    Shape: (M, n)
    codeword_angles = (
        geometry.TWO_PI * codewords.to(torch.float64) / float(alphabet_size)
        + phi
    ) % geometry.TWO_PI

    # 2. Circular distances between angle estimates and codeword angles.
    #    angle_estimates is (n,), broadcast to (1, n) against (M, n).
    dists = geometry.circular_dist(
        angle_estimates.unsqueeze(0),   # (1, n)
        codeword_angles,                # (M, n)
    )  # -> (M, n), values in [0, pi]

    # 3. Apply scoring function
    if scoring == "linear":
        # f(d) = d — simple circular distance sum
        scores_per_position = dists
    elif scoring == "log":
        # f(d) = -log(1 - d / d_max) — capacity-achieving scorer
        # Equivalent to the legacy: log(max(1 - d/pi, eps)) negated.
        # Lower is better, so we use -log(1 - d/d_max) directly.
        if d_max is None:
            d_max = math.pi
        eps = 1e-12
        scores_per_position = -torch.log(
            torch.clamp(1.0 - dists / d_max, min=eps)
        )
    else:
        raise ValueError(
            f"Unknown scoring function: {scoring!r}. "
            f"Expected 'linear' or 'log'."
        )

    # 4. Sum over token positions -> (M,)
    scores = scores_per_position.sum(dim=1)

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# Full decode pipelines
# ═══════════════════════════════════════════════════════════════════════════


def decode_message(
    tokens: Tensor,
    *,
    vocab_size: int,
    alphabet_size: int,
    num_keys: int,
    seed: int,
    codewords: Tensor,
    phi: float = 0.0,
    scoring: Literal["linear", "log"] = "log",
    config: ArcMarkConfig | None = None,
    no_watermark_threshold: float | None = None,
) -> ArcMarkDecodeResult:
    """Full decode pipeline: tokens -> symbol angles -> message.

    Convenience function that chains
    :func:`~arcmark.symbol_decoder.decode_symbol_angles` and
    :func:`score_all_messages`.

    Args:
        tokens:         1-D ``LongTensor`` of watermarked token IDs.
        vocab_size:     Total vocabulary size N.
        alphabet_size:  Code alphabet size p.
        num_keys:       Side-information alphabet size r.
        seed:           Shared secret seed.
        codewords:      All candidate codewords, shape ``(M, n)``.
        phi:            Angle offset.
        scoring:        Scoring function name.
        no_watermark_threshold: Per-token score threshold for declaring
                        text as unwatermarked.  If ``best_score / n``
                        exceeds this value, :data:`NOWM_IDX` is returned.
        config:         :class:`~arcmark.config.ArcMarkConfig` for key generation.

    Returns:
        :class:`ArcMarkDecodeResult` with decoded message index and scores.
        ``message_bits`` is ``None`` (use :func:`decode_with_code` to
        get bit-level output).
    """
    # Lazy import to avoid circular dependency at module level
    from arcmark.symbol_decoder import decode_symbol_angles

    # Step 1: symbol-level decoding — tokens + keys → angle estimates
    angle_estimates = decode_symbol_angles(
        tokens,
        vocab_size=vocab_size,
        num_keys=num_keys,
        seed=seed,
        config=config,
    )

    # Step 2: score all candidate messages
    scores = score_all_messages(
        angle_estimates,
        codewords,
        alphabet_size=alphabet_size,
        phi=phi,
        scoring=scoring,
    )

    # Step 3: decode — argmin of scores
    n = angle_estimates.shape[0]
    message_idx = int(scores.argmin().item())
    best_score = float(scores[message_idx].item())

    # Step 4 (optional): no-watermark detection
    if no_watermark_threshold is not None and n > 0:
        per_token_score = best_score / n
        if per_token_score > no_watermark_threshold:
            message_idx = NOWM_IDX

    return ArcMarkDecodeResult(
        message_idx=message_idx,
        scores=scores,
        best_score=best_score,
        message_bits=None,
    )


def decode_with_code(
    tokens: Tensor,
    *,
    vocab_size: int,
    num_keys: int,
    seed: int,
    code: ChannelCode,
    phi: float = 0.0,
    scoring: Literal["linear", "log"] = "log",
    config: ArcMarkConfig | None = None,
    no_watermark_threshold: float | None = None,
) -> ArcMarkDecodeResult:
    """Full decode pipeline using a :class:`~arcmark.coding.ChannelCode`.

    Convenience wrapper around :func:`decode_message` that:

    1. Extracts ``codebook`` and ``alphabet_size`` from the code object.
    2. Fills ``message_bits`` in the result via
       :meth:`~arcmark.coding.ChannelCode.int_to_bits`.

    Args:
        tokens:     1-D ``LongTensor`` of watermarked token IDs.
        vocab_size: Total vocabulary size N.
        num_keys:   Side-information alphabet size r.
        seed:       Shared secret seed.
        code:       A :class:`~arcmark.coding.ChannelCode` instance
                    (e.g. :class:`~arcmark.coding.RandomLinearCode`).
        phi:        Angle offset.
        scoring:    Scoring function name.
        config:     :class:`~arcmark.config.ArcMarkConfig` for key generation.
        no_watermark_threshold: Per-token score threshold for declaring
                    text as unwatermarked.  See :func:`decode_message`.

    Returns:
        :class:`ArcMarkDecodeResult` with ``message_bits`` filled.
        When no watermark is detected, ``message_idx`` is :data:`NOWM_IDX`
        and ``message_bits`` is ``None``.
    """
    result = decode_message(
        tokens,
        vocab_size=vocab_size,
        alphabet_size=code.alphabet_size,
        num_keys=num_keys,
        seed=seed,
        codewords=code.codebook,
        phi=phi,
        scoring=scoring,
        config=config,
        no_watermark_threshold=no_watermark_threshold,
    )
    if result.message_idx != NOWM_IDX:
        result.message_bits = code.int_to_bits(result.message_idx)
    return result
