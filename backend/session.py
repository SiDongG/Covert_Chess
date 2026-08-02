"""DemoSession — one interactive BAM chess game.

All heavy work (GPU inference + Sinkhorn OT) runs in a background thread;
an asyncio.Queue bridges each token to the async WebSocket handler.
"""
from __future__ import annotations
import asyncio
import threading
from typing import Any, Callable, Optional

import numpy as np
import torch

from bam.lm_backend       import HFLMBackend           # ← bam package
from bam.arcmark_adapter  import ArcMarkAdapter, ArcMarkConfig
from bam.bam_encoder      import CovertEncoder
from bam.bam_decoder      import Decoder
from bam.bam_tracker      import BAMTracker, BAMConfig
from chess_engine          import ChessInterface

Sender = Callable[[dict], Any]

_DEFAULT_PROMPT_A = (
    "You are a conversational assistant mediating a user's messages. "
    "The user has given you a topic or message — write a natural, casual reply "
    "as if you are the user speaking to a friend. "
    "Keep it 2-3 sentences, conversational and flowing. "
    "Never mention chess, games, or any hidden information."
)

_DEFAULT_PROMPT_B = (
    "You are a friendly AI having a casual conversation. "
    "Reply naturally to the message you receive, continuing the topic in a "
    "warm, conversational way. Keep it 2-3 sentences. "
    "Never mention chess, games, or any hidden information."
)
_MAX_HISTORY    = 8
_MAX_NEW_TOKENS = 150


class DemoSession:
    def __init__(
        self,
        lm1: HFLMBackend,
        lm2: HFLMBackend,
        stockfish_path: str = "/usr/games/stockfish",
        bam_cfg:     Optional[BAMConfig]    = None,
        adapter_cfg: Optional[ArcMarkConfig] = None,
        gen_lock:    Optional[asyncio.Semaphore] = None,
    ) -> None:
        self.lm1 = lm1
        self.lm2 = lm2

        # IMPORTANT: adapter_cfg.p_field MUST equal bam_cfg.p_field
        cfg_bam = bam_cfg or BAMConfig(
            eps_noise_comm=0.5,
            eps_noise_conf=0.3,
            gamma_1=0.85,
            rho_ack=0.95, rho_nack=0.95,
            p_field=4,
        )
        cfg_arc = adapter_cfg or ArcMarkConfig(
            p_field=4,          # must match cfg_bam.p_field
            r_resolution=8,
            shared_seed=0xA12C,
            top_k=50,
            sinkhorn_max_iter=1000,
            sinkhorn_stop_thr=1e-4,
            sinkhorn_reg=0.2,
            sinkhorn_method="sinkhorn_log",
        )
        self.bam_cfg = cfg_bam
        self.adapter = ArcMarkAdapter(lm1, cfg_arc)

        # User→AI direction
        self.encoder_1 = CovertEncoder(lm1, self.adapter, cfg_bam)
        self.decoder_2  = Decoder(lm2, self.adapter, cfg_bam)
        # AI→user direction
        self.encoder_2 = CovertEncoder(lm2, self.adapter, cfg_bam)

        self.chess = ChessInterface(stockfish_path)
        self.history_1: list[dict] = []
        self.history_2: list[dict] = []
        self.turn_count = 0
        # System prompts — updatable at runtime via set_prompts()
        self.prompt_a = _DEFAULT_PROMPT_A
        self.prompt_b = _DEFAULT_PROMPT_B
        # Shared GPU lock — prevents concurrent model inference across sessions
        self._gen_lock = gen_lock or asyncio.Semaphore(1)

    # ── public API ───────────────────────────────────────────────────────

    async def handle_user_turn(
        self, chat: str, move_uci: str, send: Sender
    ) -> None:
        # 1. Parse move
        try:
            M = self.chess.num_legal_moves()
            m = self.chess.move_to_index(move_uci)
        except (ValueError, IndexError) as exc:
            await send({"type": "error", "msg": f"Illegal move {move_uci}: {exc}"})
            return

        await send({"type": "status",
                    "msg": f"Embedding move {move_uci} (candidate {m+1}/{M})…"})

        # 2. Arm encoder_1 + decoder_2
        _force_reset(self.encoder_1)
        self.encoder_1.queue_message(m, M)
        self.decoder_2.expect_message(M)

        # 3. Stream LLM_1's watermarked turn
        prompt_1    = self._build_prompt(self.lm1, self.history_1, chat, self.prompt_a)
        prompt_ids1 = self.lm1.encode_tensor(prompt_1)
        text_1_buf: list[str] = []

        async for tok_id, tok_str, belief in self._stream(
            lm=self.lm1,
            enc_tracker=self.encoder_1.tracker,
            dec_tracker=self.decoder_2.tracker,
            m_true=m, prompt_ids=prompt_ids1,
        ):
            text_1_buf.append(tok_str)
            msg: dict = {"type": "token", "agent": "llm1", "text": tok_str}
            if belief:
                msg["belief"] = belief
            await send(msg)

        text_1 = "".join(text_1_buf).strip()
        await send({"type": "turn_done", "agent": "llm1", "text": text_1})

        # 4. Recover decoded move
        # Adaptive stopping rules (paper §adaptive-stopping):
        #   (a) ACK fired → dec_tracker.decoded is set (normal path)
        #   (b) EOS hit without ACK → forced decode: always commit to argmax.
        #       If belief is low it will likely be a mismatch, but "?" is
        #       never correct — the argmax IS the decoder's best guess.
        dec_tracker = self.decoder_2.tracker
        decoded_idx = dec_tracker.decoded if dec_tracker else None
        via = "ack"

        if decoded_idx is None and dec_tracker is not None:
            eff     = dec_tracker.effective_pi()
            decoded_idx = int(eff.argmax())   # always commit to argmax
            via = "forced"

        decoded_uci = None
        if decoded_idx is not None:
            try:
                decoded_uci = self.chess.index_to_move(decoded_idx).uci()
            except IndexError:
                pass

        await send({"type": "decode_result", "expected": move_uci,
                    "decoded": decoded_uci, "correct": decoded_uci == move_uci,
                    "via": via})

        # 5. Apply user's move
        try:
            user_san = self.chess.push_uci(move_uci)
        except ValueError as exc:
            await send({"type": "error", "msg": str(exc)})
            return

        self.history_1.append({"role": "user",      "content": chat})
        self.history_1.append({"role": "assistant",  "content": text_1})

        if self.chess.is_game_over():
            await send({"type": "game_over", "result": self.chess.outcome(),
                        "fen": self.chess.fen()})
            return

        # 6. Stockfish reply
        await send({"type": "status", "msg": "Stockfish thinking…"})
        loop = asyncio.get_event_loop()
        engine_move, m_star = await loop.run_in_executor(
            None, self.chess.best_move_and_index
        )
        M_star = self.chess.num_legal_moves()

        await send({"type": "engine_move", "move": engine_move.uci(),
                    "san": self.chess.board.san(engine_move)})

        # 7. Arm encoder_2
        _force_reset(self.encoder_2)
        self.encoder_2.queue_message(m_star, M_star)

        # 8. Stream LLM_2's watermarked reply
        prompt_2    = self._build_prompt(self.lm2, self.history_2, text_1, self.prompt_b)
        prompt_ids2 = self.lm2.encode_tensor(prompt_2)
        text_2_buf: list[str] = []

        async for _, tok_str, _ in self._stream(
            lm=self.lm2,
            enc_tracker=self.encoder_2.tracker,
            dec_tracker=None,
            m_true=m_star, prompt_ids=prompt_ids2,
        ):
            text_2_buf.append(tok_str)
            await send({"type": "token", "agent": "llm2", "text": tok_str})

        text_2 = "".join(text_2_buf).strip()
        await send({"type": "turn_done", "agent": "llm2", "text": text_2})

        # 9. Apply engine move; send board update (client applies on Decode click)
        engine_san = self.chess.push(engine_move)
        self.history_2.append({"role": "user",      "content": text_1})
        self.history_2.append({"role": "assistant",  "content": text_2})
        self.turn_count += 1

        await send({"type": "board_update", "fen": self.chess.fen(),
                    "user_move": move_uci, "user_san": user_san,
                    "engine_move": engine_move.uci(), "engine_san": engine_san,
                    "turn": self.turn_count})

        if self.chess.is_game_over():
            await send({"type": "game_over", "result": self.chess.outcome(),
                        "fen": self.chess.fen()})

    def reset(self) -> None:
        self.chess.reset()
        self.history_1.clear()
        self.history_2.clear()
        self.turn_count = 0
        _force_reset(self.encoder_1)
        _force_reset(self.encoder_2)
        self.decoder_2.reset()

    # ── prompt ───────────────────────────────────────────────────────────

    def set_prompts(self, prompt_a: str = "", prompt_b: str = "") -> None:
        """Update system prompts at runtime (empty string keeps current value)."""
        if prompt_a.strip():
            self.prompt_a = prompt_a.strip()
        if prompt_b.strip():
            self.prompt_b = prompt_b.strip()

    def _build_prompt(
        self, lm: HFLMBackend, history: list[dict],
        new_user_msg: str, system_prompt: str
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-_MAX_HISTORY:])
        messages.append({"role": "user", "content": new_user_msg})
        return lm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── async streaming ──────────────────────────────────────────────────

    async def _stream(
        self,
        lm:          HFLMBackend,
        enc_tracker: Optional[BAMTracker],
        dec_tracker: Optional[BAMTracker],
        m_true:      Optional[int],
        prompt_ids:  torch.Tensor,
    ):
        queue: asyncio.Queue = asyncio.Queue()
        loop  = asyncio.get_event_loop()

        def _bg() -> None:
            try:
                for item in _generate_tokens(
                    lm, enc_tracker, dec_tracker,
                    m_true, prompt_ids, _MAX_NEW_TOKENS,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("__err__", str(exc), None)), loop
                ).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        thread = threading.Thread(target=_bg, daemon=True)

        async with self._gen_lock:   # ← serialises GPU: at most one generation at a time
            thread.start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                tok_id, tok_str, belief = item
                if tok_id == "__err__":
                    raise RuntimeError(tok_str)
                yield tok_id, tok_str, belief

            thread.join(timeout=10.0)


# ── module-level helpers ─────────────────────────────────────────────────

def _force_reset(enc: CovertEncoder) -> None:
    enc.tracker      = None
    enc.true_message = None


def _generate_tokens(
    lm:          HFLMBackend,
    enc_tracker: Optional[BAMTracker],
    dec_tracker: Optional[BAMTracker],
    m_true:      Optional[int],
    prompt_ids:  torch.Tensor,
    max_new_tokens: int,
):
    ids      = prompt_ids
    stop_ids: set[int] = set()
    if lm.eos_token_id is not None:
        stop_ids.add(lm.eos_token_id)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            p = lm.next_token_distribution(ids)
        p_np = p.detach().to(torch.float64).cpu().numpy()

        if enc_tracker is not None and not enc_tracker.done:
            tok_id = enc_tracker.step_encoder(p_np, m_true)
        else:
            tok_id = int(torch.multinomial(p, num_samples=1).item())

        belief: Optional[dict] = None
        if dec_tracker is not None and not dec_tracker.done:
            dec_tracker.step_decoder(tok_id)
            eff    = dec_tracker.effective_pi()
            belief = {
                "top_idx":  int(eff.argmax()),
                "top_prob": round(float(eff.max()), 4),
                "phase":    dec_tracker.phase,
                "done":     dec_tracker.done,
            }

        tok_str = lm.decode([tok_id])
        ids     = torch.cat(
            [ids, torch.tensor([[tok_id]], device=lm.device)], dim=1
        )
        yield tok_id, tok_str, belief

        if tok_id in stop_ids:
            break