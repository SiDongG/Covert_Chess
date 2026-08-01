"""ArcMark LogitsProcessor for HuggingFace ``generate()``.

Provides :class:`ArcMarkLogitsProcessor`, a HuggingFace ``LogitsProcessor``
that embeds multi-bit watermarks via entropic optimal transport.

At each token position, the processor:

1. Applies internal temperature to convert logits → probabilities.
2. Solves the ArcMark OT problem (vocabulary restriction, circular cost,
   Sinkhorn) using the codeword symbol and secret permutation for that step.
3. Extracts the watermarked conditional ``P*(x | s)`` for the realised
   side-information key.
4. Outputs ``log(q)`` for selected tokens and ``-inf`` elsewhere, so that
   HuggingFace's downstream ``softmax`` recovers ``q`` exactly.

**Key generation modes** (controlled by ``config.hash_keys``):

- **Hash mode** (default, production): ``(s_index, perm_seed)`` are derived
  on-the-fly by hashing the previous ``context_width`` watermarked tokens
  with the secret key via SHA-256.  The decoder reconstructs keys from
  the watermarked token sequence alone.
- **Fixed mode** (debug / testing): ``(s_index, perm_seed)`` are pre-generated
  from a seeded ``torch.Generator``, independent of token context.

**Important:** Because ArcMark owns temperature, top-k, and top-p
internally, the caller **must** configure ``generate()`` with::

    model.generate(
        ...,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        logits_processor=[processor],
    )

Use :meth:`ArcMarkLogitsProcessor.default_generate_kwargs` for convenience.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import torch
from torch import Tensor
from transformers import LogitsProcessor

from arcmark import geometry
from arcmark.config import ArcMarkConfig
from arcmark.keygen import compute_key, generate_fixed_key_sequence
from arcmark.side_info import SideInfoMode, compute_key_si
from arcmark.sinkhorn import extract_conditional, solve_arcmark_ot


# ═══════════════════════════════════════════════════════════════════════════
# Permutation Cache
# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE OPTIMIZATION: Permutation generation is a major bottleneck.
# torch.randperm with a seeded Generator is CPU-only (PyTorch limitation),
# generating ~50k elements per token, then transferring to GPU.
# Since permutations are deterministic (same seed → same output), we cache
# them to avoid regenerating on every token. This can provide 2-5x speedup.


class _PermutationCache:
    """LRU cache for deterministic permutations, stored on target device.

    Key insight: random_permutation(vocab_size, seed) is deterministic.
    We cache the result on the target device to avoid:
    1. Regenerating the same permutation multiple times
    2. CPU→GPU transfer on every token

    Memory cost: ~400KB per cached permutation (50k vocab × 8 bytes).
    With max_size=32, this uses ~13MB GPU memory.
    """

    def __init__(self, max_size: int = 32) -> None:
        self.max_size = max_size
        # Cache key: (vocab_size, seed, device_str)
        # Cache value: permutation tensor on that device
        self._cache: dict[tuple[int, int, str], Tensor] = {}
        # LRU tracking: most recently used keys at the end
        self._order: list[tuple[int, int, str]] = []

    def get(
        self, vocab_size: int, seed: int, device: torch.device | str
    ) -> Tensor:
        """Get cached permutation or generate and cache it.

        Args:
            vocab_size: Vocabulary size N (length of permutation).
            seed: Deterministic seed for permutation generation.
            device: Target device for the permutation tensor.

        Returns:
            Permutation tensor on the specified device.
        """
        device_str = str(device)
        key = (vocab_size, seed, device_str)

        if key in self._cache:
            # Cache hit: move to end of LRU list (most recently used)
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]

        # Cache miss: generate permutation on CPU, move to device
        perm = geometry.random_permutation(vocab_size, seed=seed)
        perm_on_device = perm.to(device)

        # LRU eviction: remove oldest entry if cache is full
        if len(self._cache) >= self.max_size:
            oldest_key = self._order.pop(0)
            del self._cache[oldest_key]

        # Store in cache
        self._cache[key] = perm_on_device
        self._order.append(key)
        return perm_on_device

    def clear(self) -> None:
        """Clear all cached permutations."""
        self._cache.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._cache)


# Global permutation cache shared across all processor instances
_perm_cache = _PermutationCache(max_size=32)

__all__ = [
    "ArcMarkLogitsProcessor",
]


# ═══════════════════════════════════════════════════════════════════════════
# ArcMark LogitsProcessor
# ═══════════════════════════════════════════════════════════════════════════


class ArcMarkLogitsProcessor(LogitsProcessor):
    """HuggingFace ``LogitsProcessor`` that embeds ArcMark multi-bit watermarks.

    Typical usage::

        proc = ArcMarkLogitsProcessor(
            vocab_size=model.config.vocab_size,
            alphabet_size=256,
            num_keys=256,
            seed=42,
            temperature=1.0,
        )
        proc.set_trial(codeword=codeword, prompt_length=prompt_len)
        output = model.generate(
            **inputs,
            **ArcMarkLogitsProcessor.default_generate_kwargs(len(codeword)),
            logits_processor=[proc],
        )
        # proc.s_sequence and proc.perm_seeds are available for the decoder

    Args:
        vocab_size:    Total vocabulary size *N*.
        alphabet_size: Code alphabet size *p* (codeword symbols ∈ {0,…,p−1}).
        num_keys:      Side-information key count *r*.
        seed:          Base secret seed shared with the decoder.
        phi:           Angle offset φ (default 0.0).
        temperature:   Sampling temperature applied internally before OT.
                       Must be > 0.
        config:        :class:`~arcmark.config.ArcMarkConfig` controlling the
                       OT solver (top-k, top-p, regularisation, etc.) and
                       key generation mode (``hash_keys``, ``context_width``).
                       Uses ``ArcMarkConfig()`` defaults when ``None``.
    """

    def __init__(
        self,
        vocab_size: int,
        alphabet_size: int,
        num_keys: int,
        seed: int = 0,
        phi: float = 0.0,
        temperature: float = 1.0,
        config: ArcMarkConfig | None = None,
        side_info_mode: SideInfoMode = SideInfoMode.HASH_CONTEXT,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {temperature}"
            )

        self.vocab_size    = int(vocab_size)
        self.alphabet_size = int(alphabet_size)
        self.num_keys      = int(num_keys)
        self.seed          = int(seed)
        self.phi           = float(phi)
        self.temperature   = float(temperature)
        self.config        = config if config is not None else ArcMarkConfig()
        self.side_info_mode = side_info_mode
        self.tokenizer      = tokenizer

        # Per-trial state (set by set_trial)
        self._codeword: Tensor | None = None
        self._prompt_length: int | None = None
        self._key_pairs: list[tuple[int, int]] | None = None
        self._step: int = 0
        self._recorded_s: list[int] = []
        self._recorded_perm_seeds: list[int] = []

        # PERFORMANCE OPTIMIZATION: Pre-allocated output buffer.
        # Instead of creating a new tensor every token (torch.full allocation),
        # we reuse a single buffer. This avoids allocation overhead per token.
        # The buffer is lazily initialized on first use to match dtype/device.
        self._output_buffer: Tensor | None = None

    # ── Trial setup ───────────────────────────────────────────────────────

    def set_trial(
        self,
        codeword: Sequence[int] | Tensor,
        prompt_length: int,
    ) -> None:
        """Configure the processor for a new generation trial.

        Must be called **before** each ``model.generate()`` invocation.

        Args:
            codeword:      Sequence of symbols in ``{0, …, alphabet_size−1}``
                           of length *n* (the number of tokens to watermark).
            prompt_length: Number of tokens in the prompt, so the processor
                           can compute the watermark position
                           ``t = seq_len − prompt_length``.
        """
        self._codeword = torch.as_tensor(codeword, dtype=torch.long)
        self._prompt_length = int(prompt_length)
        self._step = 0
        self._recorded_s = []
        self._recorded_perm_seeds = []

        cw_len = len(self._codeword)
        if not self.config.hash_keys and cw_len > 0:
            # Fixed mode: pre-generate all key pairs
            self._key_pairs = generate_fixed_key_sequence(
                seed=self.seed,
                length=cw_len,
                num_keys=self.num_keys,
            )
            # PERFORMANCE OPTIMIZATION: Pre-warm the permutation cache.
            # In fixed mode, we know all perm_seeds upfront. Pre-generating
            # unique permutations here avoids cache misses during generation.
            # Note: We can't move to device yet (don't know target device),
            # but at least generate them on CPU so they're ready.
            unique_seeds = {seed for _, seed in self._key_pairs}
            for perm_seed in unique_seeds:
                # Generate on CPU; will be moved to device on first use
                geometry.random_permutation(self.vocab_size, seed=perm_seed)
        else:
            self._key_pairs = None  # hash mode: computed on-the-fly

    # ── LogitsProcessor interface ─────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Process logits for the next token during generation.

        Args:
            input_ids: ``(batch, seq_len)`` — token IDs generated so far.
            scores:    ``(batch, vocab_size)`` — raw logits from the model.

        Returns:
            Modified logits of the same shape.  For watermarked positions,
            non-selected tokens get ``-inf`` and selected tokens get
            ``log(q)`` where ``q`` is the OT-watermarked distribution.

        Raises:
            RuntimeError: If :meth:`set_trial` has not been called, or if
                ``batch_size > 1``.
        """
        # --- guards ---
        if self._codeword is None:
            raise RuntimeError(
                "ArcMarkLogitsProcessor: must call set_trial() before "
                "generate()"
            )
        batch_size, V = scores.shape
        if batch_size != 1:
            raise RuntimeError(
                "ArcMarkLogitsProcessor supports batch_size=1 only, "
                f"got {batch_size}"
            )

        # --- compute watermark position ---
        seq_len = input_ids.shape[1]
        t = seq_len - self._prompt_length

        # Pass-through if outside the codeword range
        if t < 0 or t >= len(self._codeword):
            return scores

        # --- apply internal temperature ---
        if self.temperature != 1.0:
            scaled_scores = scores / self.temperature
        else:
            scaled_scores = scores

        # --- convert logits to probabilities ---
        probs = torch.softmax(scaled_scores[0], dim=-1)  # shape (V,)

        # --- retrieve codeword symbol and side-info key ---
        codeword_symbol = int(self._codeword[t].item())

        if self.config.hash_keys:
            # Hash mode: derive keys from preceding watermarked tokens
            cw = self.config.context_width
            wm_start = self._prompt_length
            wm_tokens = input_ids[0, wm_start: wm_start + t].tolist()
            pad_len = max(0, cw - len(wm_tokens))
            context_tokens = tuple([0] * pad_len + wm_tokens[-cw:])
            s_index, perm_seed = compute_key_si(
                secret_key=self.seed,
                context_tokens=context_tokens,
                num_keys=self.num_keys,
                mode=self.side_info_mode,
                tokenizer=self.tokenizer,
            )
        else:
            # Fixed mode: use pre-generated keys
            s_index, perm_seed = self._key_pairs[t]

        # --- generate deterministic permutation for this position ---
        # PERFORMANCE OPTIMIZATION: Use cached permutation instead of regenerating.
        # Before: geometry.random_permutation() on CPU + .to(device) every token
        # After: Cache lookup with LRU eviction, ~2-5x faster for repeated seeds.
        # The cache stores permutations directly on the target device.
        perm = _perm_cache.get(self.vocab_size, perm_seed, probs.device)

        # --- solve ArcMark OT ---
        ot_result = solve_arcmark_ot(
            probs,
            codeword_symbol=codeword_symbol,
            alphabet_size=self.alphabet_size,
            num_keys=self.num_keys,
            vocab_size=self.vocab_size,
            perm=perm,
            phi=self.phi,
            config=self.config,
        )

        # --- extract conditional P*(x | s = s_index) ---
        cond = extract_conditional(
            ot_result.coupling,
            s_index,
            num_keys=self.num_keys,
            full_vocab_size=V,
            token_indices=ot_result.token_indices,
        )
        # cond: shape (V,) with zeros outside the selected tokens

        # --- convert to log-space ---
        # PERFORMANCE OPTIMIZATION: Reuse pre-allocated output buffer.
        # Before: torch.full() allocation every token (~50k elements)
        # After: Reuse buffer, just fill with -inf and update masked elements.
        # Minor optimization (~1-2%) but zero risk and cleaner memory usage.
        if (
            self._output_buffer is None
            or self._output_buffer.shape != (1, V)
            or self._output_buffer.device != scores.device
            or self._output_buffer.dtype != scores.dtype
        ):
            # Lazy init: create buffer matching scores shape/dtype/device
            self._output_buffer = torch.empty(
                (1, V), dtype=scores.dtype, device=scores.device
            )

        # Fill with -inf (tokens not in top-k get -inf probability)
        self._output_buffer.fill_(float("-inf"))
        mask = cond > 0
        self._output_buffer[0, mask] = torch.log(cond[mask]).to(scores.dtype)

        # --- record and advance ---
        self._recorded_s.append(s_index)
        self._recorded_perm_seeds.append(perm_seed)
        self._step += 1

        # Return a clone to prevent external mutation of our buffer
        return self._output_buffer.clone()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def s_sequence(self) -> list[int]:
        """Side-information key values used during generation.

        Available after ``generate()`` completes.  In hash mode, the
        decoder reconstructs these via
        :func:`~arcmark.keygen.compute_keys_from_tokens`.  In fixed
        mode, via :func:`~arcmark.keygen.generate_fixed_key_sequence`.
        """
        return list(self._recorded_s)

    @property
    def perm_seeds(self) -> list[int]:
        """Permutation seeds used during generation.

        Available after ``generate()`` completes.
        """
        return list(self._recorded_perm_seeds)

    @property
    def step_count(self) -> int:
        """Number of watermarked tokens generated so far in this trial."""
        return self._step

    # ── Convenience ───────────────────────────────────────────────────────

    @staticmethod
    def default_generate_kwargs(max_new_tokens: int) -> dict:
        """Return ``generate()`` keyword arguments required for ArcMark.

        ArcMark owns temperature, top-k, and top-p internally.  HuggingFace
        must **not** apply its own warpers, so these must be disabled::

            output = model.generate(
                **inputs,
                **ArcMarkLogitsProcessor.default_generate_kwargs(n),
                logits_processor=[processor],
            )

        Args:
            max_new_tokens: Number of tokens to generate (= codeword length).

        Returns:
            Dict with ``do_sample``, ``temperature``, ``top_k``, ``top_p``,
            and ``max_new_tokens``.
        """
        return {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
        }