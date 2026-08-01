"""ArcMark LogitsProcessor using hash-column codebook.

Extends ArcMarkLogitsProcessor to derive each codeword column on-the-fly
from a hash of the context tokens, rather than using a fixed pre-built
generator matrix.

At each position t:
    g_t          = hash_to_vector(secret_key, context_tokens_t)
    symbol_t     = m_vec · g_t  mod p
    target_angle = 2π · symbol_t / p  +  2π · s_index / r

The side-info shift s_index is computed exactly as before — unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import LogitsProcessor

from arcmark import geometry
from arcmark.config import ArcMarkConfig
from arcmark.side_info import SideInfoMode, compute_key_si
from arcmark.sinkhorn import extract_conditional, solve_arcmark_ot
from arcmark.processor import _perm_cache
from arcmark.hash_column_code import HashColumnCode, hash_to_vector

import numpy as np


class HashColumnLogitsProcessor(LogitsProcessor):
    """ArcMark processor with hash-derived codebook columns.

    Args:
        vocab_size:      Model vocabulary size.
        k_bits:          Number of watermark bits.
        p:               Alphabet size.
        num_keys:        Side-information cardinality r.
        seed:            Shared integer secret.
        message_idx:     Message to embed (set via set_trial).
        config:          ArcMarkConfig.
        side_info_mode:  Key derivation mode.
        tokenizer:       HuggingFace tokenizer.
    """

    def __init__(
        self,
        vocab_size: int,
        k_bits: int,
        p: int,
        num_keys: int,
        seed: int,
        temperature: float = 1.0,
        config: ArcMarkConfig | None = None,
        side_info_mode: SideInfoMode = SideInfoMode.HASH_CONTEXT,
        tokenizer=None,
    ) -> None:
        self.vocab_size      = vocab_size
        self.k_bits          = k_bits
        self.p               = p
        self.num_keys        = num_keys
        self.seed            = seed
        self.temperature     = temperature
        self.config          = config if config is not None else ArcMarkConfig()
        self.side_info_mode  = side_info_mode
        self.tokenizer       = tokenizer

        self._message_idx:   int | None  = None
        self._m_vec:         np.ndarray | None = None
        self._prompt_length: int | None  = None
        self._d = max(1, int(np.ceil(k_bits * np.log(2) / np.log(p))))

    def set_trial(self, message_idx: int, prompt_length: int) -> None:
        """Set the message and prompt length for one generation trial."""
        import math
        self._message_idx  = message_idx
        self._prompt_length = int(prompt_length)

        # Precompute m_vec (base-p digits of message_idx)
        m = np.empty(self._d, dtype=np.int32)
        val = int(message_idx)
        for j in range(self._d):
            m[j] = val % self.p
            val //= self.p
        self._m_vec = m.astype(np.int64)

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self._message_idx is None:
            raise RuntimeError("Must call set_trial() before generate()")

        batch_size, V = scores.shape
        seq_len = input_ids.shape[1]
        t = seq_len - self._prompt_length

        if t < 0:
            return scores

        # ── Compute context tokens ────────────────────────────────────────
        cw       = self.config.context_width
        wm_start = self._prompt_length
        wm_tokens = input_ids[0, wm_start: wm_start + t].tolist()
        pad_len   = max(0, cw - len(wm_tokens))
        context_tokens = tuple([0] * pad_len + wm_tokens[-cw:])

        # ── Derive hash column g_t and compute codeword symbol ────────────
        g_t    = hash_to_vector(self.seed, context_tokens, self._d, self.p)
        symbol = int(np.dot(self._m_vec, g_t.astype(np.int64)) % self.p)

        # ── Compute side-info key (unchanged) ─────────────────────────────
        s_index, perm_seed = compute_key_si(
            secret_key=self.seed,
            context_tokens=context_tokens,
            num_keys=self.num_keys,
            mode=self.side_info_mode,
            tokenizer=self.tokenizer,
        )

        # ── Apply temperature ─────────────────────────────────────────────
        if self.temperature != 1.0:
            scaled = scores / self.temperature
        else:
            scaled = scores
        probs = torch.softmax(scaled[0], dim=-1)

        # ── Permutation ───────────────────────────────────────────────────
        perm = _perm_cache.get(V, perm_seed, probs.device)

        # ── OT solve ─────────────────────────────────────────────────────
        ot_result = solve_arcmark_ot(
            probs,
            codeword_symbol=symbol,
            alphabet_size=self.p,
            num_keys=self.num_keys,
            vocab_size=V,
            perm=perm,
            phi=0.0,
            config=self.config,
        )

        # ── Extract conditional P*(x | s = s_index) ───────────────────────
        cond = extract_conditional(
            ot_result.coupling,
            s_index,
            num_keys=self.num_keys,
            full_vocab_size=V,
            token_indices=ot_result.token_indices,
        )

        # ── Build output logits ───────────────────────────────────────────
        out = torch.full((1, V), float("-inf"),
                         dtype=scores.dtype, device=scores.device)
        mask = cond > 0
        out[0, mask] = torch.log(cond[mask]).to(scores.dtype)
        return out
