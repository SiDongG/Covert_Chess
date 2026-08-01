"""Experimental metrics for ArcMark watermark evaluation.

Provides functions and container dataclasses for computing:

- **Success rate**: Fraction of trials with exact message recovery.
- **Bit error rate (BER)**: Fraction of incorrect bits in decoded messages.
- **Codeword symbol error rate**: Fraction of positions where decoded and
  true codeword symbols differ.
- **Perplexity**: Language-model quality metric for watermarked text.
- **Standard error of the mean (SEM)**: Uncertainty estimates for all metrics.
- **ExperimentMetrics**: Aggregated container for a batch of trials.

All scalar-returning functions return plain Python floats for easy
serialisation. Accepts plain Python lists, PyTorch tensors, or NumPy arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

__all__ = [
    "ExperimentMetrics",
    "TrialResult",
    "aggregate_trials",
    "compute_ber",
    "compute_ber_xor",
    "compute_codeword_error_rate",
    "compute_perplexity",
    "compute_sem",
    "compute_success_rate",
]


# ═══════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrialResult:
    """Result of a single watermark trial.

    Attributes:
        sent_idx:       True message index.
        recv_idx:       Decoded message index.
        sent_bits:      True message as bit tensor, shape ``(k,)``.
        recv_bits:      Decoded message as bit tensor, shape ``(k,)``.
        sent_codeword:  True codeword, shape ``(n,)``.
        recv_codeword:  Decoded codeword, shape ``(n,)``.
        perplexity:     Per-trial perplexity (``None`` if not computed).
    """

    sent_idx: int
    recv_idx: int
    sent_bits: Tensor
    recv_bits: Tensor
    sent_codeword: Tensor
    recv_codeword: Tensor
    perplexity: float | None = None


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for a batch of watermark trials.

    All rate fields are in [0, 1]. SEM fields are non-negative.

    Attributes:
        num_trials:              Total number of trials.
        success_rate:            Fraction of exact message recovery.
        success_rate_sem:        SEM of success rate.
        bit_error_rate:          Mean fraction of incorrect message bits.
        bit_error_rate_sem:      SEM of BER.
        codeword_error_rate:     Mean fraction of incorrect codeword symbols.
        codeword_error_rate_sem: SEM of codeword error rate.
        perplexity_mean:         Mean perplexity across trials
                                 (``None`` if not computed).
        perplexity_sem:          SEM of perplexity (``None`` if not computed).
        per_trial:               List of individual :class:`TrialResult` objects.
    """

    num_trials: int
    success_rate: float
    success_rate_sem: float
    bit_error_rate: float
    bit_error_rate_sem: float
    codeword_error_rate: float
    codeword_error_rate_sem: float
    perplexity_mean: float | None = None
    perplexity_sem: float | None = None
    per_trial: list[TrialResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Primitive metric functions
# ═══════════════════════════════════════════════════════════════════════════


def compute_sem(values: Sequence[float]) -> float:
    """Standard error of the mean.

    Args:
        values: Sequence of scalar observations.

    Returns:
        ``std(values, ddof=1) / sqrt(n)``.
        Returns 0.0 if ``n <= 1``.
    """
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance / n)


def compute_success_rate(
    sent_indices: Sequence[int],
    recv_indices: Sequence[int],
) -> tuple[float, float]:
    """Fraction of trials with exact message recovery.

    Args:
        sent_indices: True message indices.
        recv_indices: Decoded message indices.

    Returns:
        ``(success_rate, success_rate_sem)`` tuple.

    Raises:
        ValueError: If the two sequences have different lengths or are empty.
    """
    if len(sent_indices) != len(recv_indices):
        raise ValueError(
            f"Length mismatch: {len(sent_indices)} vs {len(recv_indices)}"
        )
    if len(sent_indices) == 0:
        raise ValueError("Empty sequences")
    successes = [1.0 if s == r else 0.0 for s, r in zip(sent_indices, recv_indices)]
    rate = sum(successes) / len(successes)
    sem = compute_sem(successes)
    return rate, sem


def compute_ber(
    sent_bits: Tensor,
    recv_bits: Tensor,
) -> float:
    """Bit error rate for a single trial (element-wise comparison).

    Args:
        sent_bits: True bits, shape ``(k,)``.
        recv_bits: Decoded bits, shape ``(k,)``.

    Returns:
        Fraction of differing bits in [0, 1].

    Raises:
        ValueError: If the two tensors have different lengths or are empty.
    """
    if sent_bits.shape != recv_bits.shape:
        raise ValueError(
            f"Shape mismatch: {sent_bits.shape} vs {recv_bits.shape}"
        )
    k = sent_bits.numel()
    if k == 0:
        raise ValueError("Empty bit tensors")
    return float((sent_bits != recv_bits).sum().item()) / k


def compute_ber_xor(sent_idx: int, recv_idx: int, k_bits: int) -> float:
    """Bit error rate via XOR (legacy method).

    Computes the number of differing bits between the binary representations
    of ``sent_idx`` and ``recv_idx``, divided by ``k_bits``.

    This is mathematically equivalent to :func:`compute_ber` when
    ``int_to_bits`` uses MSB-first encoding (as in
    :class:`~arcmark.coding.ChannelCode`). Both methods are provided so
    users can verify equivalence.

    Args:
        sent_idx: True message index.
        recv_idx: Decoded message index.
        k_bits:   Number of message bits.

    Returns:
        Fraction of differing bits in [0, 1].

    Raises:
        ValueError: If ``k_bits < 1``.
    """
    if k_bits < 1:
        raise ValueError(f"k_bits must be >= 1, got {k_bits}")
    xor_val = sent_idx ^ recv_idx
    differing = bin(xor_val).count("1")
    return differing / k_bits


def compute_codeword_error_rate(
    sent_codeword: Tensor,
    recv_codeword: Tensor,
) -> float:
    """Fraction of codeword positions where symbols differ.

    This metric operates at the *symbol* level (before bit conversion),
    measuring how many of the n codeword positions have the wrong symbol.
    For a code with alphabet size p, a single symbol error could flip
    up to log2(p) bits.

    Args:
        sent_codeword: True codeword, shape ``(n,)``.
        recv_codeword: Decoded codeword, shape ``(n,)``.

    Returns:
        Symbol error rate in [0, 1].

    Raises:
        ValueError: If the two tensors have different lengths or are empty.
    """
    if sent_codeword.shape != recv_codeword.shape:
        raise ValueError(
            f"Shape mismatch: {sent_codeword.shape} vs {recv_codeword.shape}"
        )
    n = sent_codeword.numel()
    if n == 0:
        raise ValueError("Empty codeword tensors")
    return float((sent_codeword != recv_codeword).sum().item()) / n


# ═══════════════════════════════════════════════════════════════════════════
# Perplexity
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def compute_perplexity(
    model,
    input_ids: Tensor,
    prompt_length: int,
) -> float:
    """Compute perplexity of generated tokens given a prompt.

    Runs a single forward pass over the full sequence, then computes
    the negative log-likelihood only over the generated (post-prompt)
    tokens.  Perplexity = exp(NLL).

    This follows the standard definition used in the legacy
    ``LLM_pipeline/perplexity.py`` but is self-contained (no separate
    model wrapper needed).

    Args:
        model:          HuggingFace causal LM (already on the correct device).
        input_ids:      Full token sequence including prompt, shape ``(1, L)``
                        or ``(L,)``.
        prompt_length:  Number of tokens in the prompt prefix.

    Returns:
        Perplexity (``float >= 1.0``). Returns ``float('inf')`` if there
        are no generated tokens to evaluate.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # Move to model device if needed
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    seq_len = input_ids.shape[1]
    if prompt_length >= seq_len:
        return float("inf")

    outputs = model(input_ids=input_ids)
    # logits shape: (1, L, V)
    # For generated tokens at positions [prompt_length, ..., L-1],
    # the predictions come from logits at positions [prompt_length-1, ..., L-2]
    logits = outputs.logits[0, prompt_length - 1 : -1, :]  # (n_gen, V)
    target_ids = input_ids[0, prompt_length:]                # (n_gen,)

    n_gen = target_ids.shape[0]
    if n_gen == 0:
        return float("inf")

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(
        -1, target_ids.unsqueeze(-1),
    ).squeeze(-1)  # (n_gen,)

    nll = -token_log_probs.mean().item()
    return math.exp(nll)


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def aggregate_trials(trials: list[TrialResult]) -> ExperimentMetrics:
    """Compute aggregate metrics from a list of trial results.

    Args:
        trials: List of :class:`TrialResult` objects.

    Returns:
        :class:`ExperimentMetrics` with all fields populated.

    Raises:
        ValueError: If ``trials`` is empty.
    """
    if not trials:
        raise ValueError("No trials to aggregate")

    n = len(trials)

    # Success rate (works for both watermarked and unwatermarked trials:
    # for unwatermarked trials sent_idx == recv_idx == NOWM_IDX on success)
    successes = [1.0 if t.sent_idx == t.recv_idx else 0.0 for t in trials]
    success_rate = sum(successes) / n
    success_rate_sem = compute_sem(successes)

    # Bit error rate (element-wise per trial)
    # Skip trials where bits are None (e.g. unwatermarked trials)
    bers = [
        compute_ber(t.sent_bits, t.recv_bits)
        for t in trials
        if t.sent_bits is not None and t.recv_bits is not None
    ]
    ber_mean = sum(bers) / len(bers) if bers else 0.0
    ber_sem = compute_sem(bers) if bers else 0.0

    # Codeword symbol error rate
    # Skip trials where codewords are None (e.g. unwatermarked trials)
    cers = [
        compute_codeword_error_rate(t.sent_codeword, t.recv_codeword)
        for t in trials
        if t.sent_codeword is not None and t.recv_codeword is not None
    ]
    cer_mean = sum(cers) / len(cers) if cers else 0.0
    cer_sem = compute_sem(cers) if cers else 0.0

    # Perplexity (optional)
    ppl_values = [t.perplexity for t in trials if t.perplexity is not None]
    if ppl_values:
        ppl_mean = sum(ppl_values) / len(ppl_values)
        ppl_sem = compute_sem(ppl_values)
    else:
        ppl_mean = None
        ppl_sem = None

    return ExperimentMetrics(
        num_trials=n,
        success_rate=success_rate,
        success_rate_sem=success_rate_sem,
        bit_error_rate=ber_mean,
        bit_error_rate_sem=ber_sem,
        codeword_error_rate=cer_mean,
        codeword_error_rate_sem=cer_sem,
        perplexity_mean=ppl_mean,
        perplexity_sem=ppl_sem,
        per_trial=trials,
    )
