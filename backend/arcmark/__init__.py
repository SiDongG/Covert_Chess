"""ArcMark: Multi-bit LLM Watermark via Optimal Transport."""

from arcmark.coding import ChannelCode, RandomLinearCode
from arcmark.config import ArcMarkConfig
from arcmark.keygen import (
    compute_key,
    compute_keys_from_tokens,
    generate_fixed_key_sequence,
)
from arcmark.message_decoder import (
    NOWM_IDX,
    ArcMarkDecodeResult,
    decode_message,
    decode_with_code,
    score_all_messages,
)
from arcmark.metrics import (
    ExperimentMetrics,
    TrialResult,
    aggregate_trials,
    compute_ber,
    compute_ber_xor,
    compute_codeword_error_rate,
    compute_perplexity,
    compute_sem,
    compute_success_rate,
)
from arcmark.processor import ArcMarkLogitsProcessor
from arcmark.sinkhorn import (
    ArcMarkOTResult,
    extract_conditional,
    restrict_vocab,
    solve_arcmark_ot,
    solve_ot,
)
from arcmark.symbol_decoder import decode_symbol_angle, decode_symbol_angles

__all__ = [
    "ArcMarkConfig",
    "ArcMarkDecodeResult",
    "ArcMarkLogitsProcessor",
    "ArcMarkOTResult",
    "ChannelCode",
    "NOWM_IDX",
    "ExperimentMetrics",
    "RandomLinearCode",
    "TrialResult",
    "aggregate_trials",
    "compute_ber",
    "compute_ber_xor",
    "compute_codeword_error_rate",
    "compute_key",
    "compute_keys_from_tokens",
    "compute_perplexity",
    "compute_sem",
    "compute_success_rate",
    "decode_message",
    "decode_symbol_angle",
    "decode_symbol_angles",
    "decode_with_code",
    "extract_conditional",
    "generate_fixed_key_sequence",
    "restrict_vocab",
    "score_all_messages",
    "solve_arcmark_ot",
    "solve_ot",
]
