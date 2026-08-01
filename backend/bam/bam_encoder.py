"""Burnashev-ArcMark encoder. Replaces fixed-length Encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .arcmark_adapter import ArcMarkAdapter
from .bam_tracker import BAMTracker, BAMConfig
from .lm_backend import HFLMBackend


# Tokens that mark a natural sentence boundary, used to let the cover text
# finish its current thought once the covert message has completed instead of
# truncating mid-sentence. These are decoded-text heuristics; see
# _looks_like_sentence_end below for the actual test (which is tokenizer-
# agnostic — it inspects the decoded string rather than hard-coding token ids).
_SENTENCE_END_CHARS = (".", "!", "?", "\n")


class CovertEncoder:
    def __init__(self, lm: HFLMBackend, adapter: ArcMarkAdapter, bam_cfg: BAMConfig):
        self.lm = lm
        self.adapter = adapter
        self.bam_cfg = bam_cfg
        self.tracker: Optional[BAMTracker] = None
        self.true_message: Optional[int] = None
        self._t_offset = 0   # global token index across transmissions
        self._on_event: Optional[Callable[[str, dict], None]] = None
        self.last_covert_tokens = 0
        # Average Shannon entropy (in BITS) of the LM next-token distribution,
        # averaged over the effective embedding positions of the LAST turn.
        # 0.0 when the turn embedded no covert tokens.
        self.last_avg_entropy_bits = 0.0

    # --- payload management ---
    def queue_message(self, m: int, M: int,
                      on_event: Optional[Callable[[str, dict], None]] = None) -> None:
        """Queue message m for transmission. M is the size of the message
        alphabet for this transmission (legal_moves + 1 for STOP)."""
        if self.has_pending():
            raise RuntimeError("a covert message is still in flight")
        self._on_event = on_event
        self.true_message = m

        def sample_from(probs):
            return int(np.random.choice(self.lm.vocab_size, p=probs))

        self.tracker = BAMTracker(
            M=M, cfg=self.bam_cfg,
            channel_angle=self.adapter.channel_angle,
            coupled_dist=self.adapter._solve_ot_coupling_array,
            angle_of_token=self.adapter._angle_of_observed_token,
            phi=self.adapter.phi,
            sample_from=sample_from,
            t_offset=0,
            on_event=on_event,
        )

    def has_pending(self) -> bool:
        return self.tracker is not None and not self.tracker.done

    def clear_done(self):
        if self.tracker is not None and self.tracker.done:
            # DON'T advance _t_offset; reset to 0 for each new transmission
            self.tracker = None
            self.true_message = None

    def requeue_pending(self, M: int) -> None:
        """Reset tracker for retransmission (used on receiver-side corruption)."""
        if self.tracker is None:
            return
        m, ev = self.true_message, self._on_event
        self.tracker = None
        self.queue_message(m, M, on_event=ev)

    # --- helpers ---
    def _looks_like_sentence_end(self, token_ids: list[int]) -> bool:
        """True if the decoded text so far ends at a natural sentence boundary.

        Decoding the whole suffix each call is cheap relative to a forward pass,
        and it sidesteps tokenizer-specific punctuation token ids (many BPE
        vocabs glue punctuation to the preceding word, so checking token ids
        alone misses 'word.' style tokens).
        """
        if not token_ids:
            return False
        text = self.lm.decode(token_ids).rstrip()
        return text.endswith(_SENTENCE_END_CHARS)

    @staticmethod
    def _entropy_bits(p_np: np.ndarray) -> float:
        """Shannon entropy (bits) of a probability vector. Robust to zeros and
        un-normalised input."""
        p = np.asarray(p_np, dtype=np.float64)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return 0.0
        p = p / s
        nz = p[p > 0]
        # H = -Σ p log2 p  (abs guards against -0.0 from float rounding)
        return float(abs(-(nz * np.log2(nz)).sum()))

    # --- generation ---
    @torch.no_grad()
    def generate_turn(self, prompt_text: str, max_new_tokens: int = 96,
                      plain_stop_token_ids: Optional[list[int]] = None,
                      finish_sentence_max: int = 32,
                      ) -> tuple[str, list[int]]:
        """Generate one turn of cover text.

        Behavior:
          * While a covert message is pending, tokens are produced by the BAM
            tracker (channel-coupled sampling). If the LM naturally emits a
            stop token (EOS) during this phase, the turn ends here and the
            message resumes on a later turn — the tracker is NOT cleared.
          * Once the message completes mid-turn (tracker.done flips), generation
            switches to the LM's own sampling and continues NATURALLY until the
            model emits EOS (or the max_new_tokens ceiling). The turn is not cut
            at the first sentence boundary after embedding — that ended turns
            prematurely when a message embedded in only a few tokens.
          * With no pending message, the whole turn is plain LM sampling.

        Side effects: sets self.last_covert_tokens (effective embedding token
        count this turn) and self.last_avg_entropy_bits (mean entropy in bits
        of the LM distribution over those embedding positions only).
        """
        ids = self.lm.encode_tensor(prompt_text)
        emitted: list[int] = []
        plain_stop = set(plain_stop_token_ids or [])
        if self.lm.eos_token_id is not None:
            plain_stop.add(self.lm.eos_token_id)

        n_covert = 0
        entropy_sum_bits = 0.0   # accumulates entropy at embedding positions only

        for _ in range(max_new_tokens):
            p = self.lm.next_token_distribution(ids)
            p_np = p.detach().to(torch.float64).cpu().numpy()

            if self.has_pending():
                # Covert phase: channel decides the token value. This is an
                # "effective embedding position" — record the LM distribution's
                # entropy here (the carrier's available randomness).
                entropy_sum_bits += self._entropy_bits(p_np)
                tok = self.tracker.step_encoder(p_np, self.true_message)
                n_covert += 1
            else:
                # Plain phase (either never had a message, or it just finished).
                tok = int(torch.multinomial(p, num_samples=1).item())

            emitted.append(tok)
            ids = torch.cat([ids, torch.tensor([[tok]], device=self.lm.device)], dim=1)

            # EOS / explicit stop always ends the turn. If this fired during the
            # covert phase, the message simply continues next turn (tracker kept).
            # After the message has completed, this is the natural end of the
            # turn — the model decides when to stop, exactly as in normal
            # generation.
            if tok in plain_stop:
                break

            # No sentence-boundary truncation. Once the covert message completes
            # (has_pending() becomes False), the loop simply keeps sampling from
            # the base distribution (the `else` branch above) and ends naturally
            # when the model emits EOS, or at the max_new_tokens ceiling. We do
            # NOT cut the turn at the first period after embedding — doing so
            # ended turns prematurely when a message embedded in few tokens.

        self.last_covert_tokens = n_covert
        self.last_avg_entropy_bits = (entropy_sum_bits / n_covert) if n_covert else 0.0
        return self.lm.decode(emitted), emitted