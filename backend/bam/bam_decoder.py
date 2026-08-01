"""Burnashev-ArcMark decoder. Replaces fixed-length Decoder."""

from __future__ import annotations

from typing import Callable, Optional

from .arcmark_adapter import ArcMarkAdapter
from .bam_tracker import BAMTracker, BAMConfig
from .lm_backend import HFLMBackend


class Decoder:
    def __init__(self, lm: HFLMBackend, adapter: ArcMarkAdapter, bam_cfg: BAMConfig):
        self.lm = lm
        self.adapter = adapter
        self.bam_cfg = bam_cfg
        self.tracker: Optional[BAMTracker] = None
        self._t_offset = 0
        self._on_event: Optional[Callable[[str, dict], None]] = None

    def expect_message(self, M: int,
                       on_event: Optional[Callable[[str, dict], None]] = None) -> None:
        """Initialize tracker for the next incoming message. M from current legal moves."""
        self._on_event = on_event
        self.tracker = BAMTracker(
            M=M, cfg=self.bam_cfg,
            channel_angle=self.adapter.channel_angle,
            coupled_dist=self.adapter._solve_ot_coupling_array,
            angle_of_token=self.adapter._angle_of_observed_token,
            phi=self.adapter.phi,     # NEW
            sample_from=None,
            t_offset=0,
            on_event=on_event,
        )

    def consume_turn_ids(self, ids):
        if self.tracker is None:
            raise RuntimeError(...)
        for tok in ids:
            if self.tracker.done:
                break
            self.tracker.step_decoder(tok)
        if self.tracker.done:
            m = self.tracker.decoded
            # DON'T advance _t_offset
            self.tracker = None
            return m
        return None

    def reset(self):
        if self.tracker is not None:
            # DON'T advance _t_offset
            self.tracker = None