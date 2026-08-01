"""
Chess board + Stockfish interface for the BAM demo.

Key design constraint
---------------------
BAM encodes a message as an index into the legal-move list.  Both the encoder
(server, LLM_1 side) and decoder (server, LLM_2 side) must produce IDENTICAL
move lists for every position, because the index m must be consistent.

We achieve this by sorting legal moves lexicographically by UCI string before
indexing.  python-chess's `board.legal_moves` generator order is deterministic
for a given position, but explicit sorting is safer and documents the contract.
"""
from __future__ import annotations

from typing import Optional

import chess
import chess.engine


class ChessInterface:
    """python-chess board + Stockfish subprocess."""

    def __init__(
        self,
        stockfish_path: str = "/usr/games/stockfish",
        skill_level: int = 5,          # 0–20; 5 ≈ 1400 Elo
        move_time_s: float = 0.3,
    ) -> None:
        self.board = chess.Board()
        self.move_time_s = move_time_s
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._engine.configure({"Skill Level": skill_level})
        self._history: list[dict] = []          # [{uci, san, color}, ...]

    # ------------------------------------------------------------------
    # Move alphabet  (M = number of legal moves; m = index)
    # ------------------------------------------------------------------

    def legal_moves(self) -> list[chess.Move]:
        """Stable, shared-secret–independent legal move list."""
        return sorted(self.board.legal_moves, key=lambda mv: mv.uci())

    def num_legal_moves(self) -> int:
        return len(self.legal_moves())

    def move_to_index(self, uci: str) -> int:
        """Raise ValueError if move is illegal."""
        move = chess.Move.from_uci(uci)
        return self.legal_moves().index(move)

    def index_to_move(self, idx: int) -> chess.Move:
        return self.legal_moves()[idx]

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------

    def best_move(self) -> chess.Move:
        """Ask Stockfish for the best move in the current position."""
        result = self._engine.play(
            self.board,
            chess.engine.Limit(time=self.move_time_s),
        )
        assert result.move is not None, "Stockfish returned no move"
        return result.move

    def best_move_and_index(self) -> tuple[chess.Move, int]:
        """(move, index) before applying the move to the board."""
        legal = self.legal_moves()
        move  = self.best_move()
        return move, legal.index(move)

    # ------------------------------------------------------------------
    # Board manipulation
    # ------------------------------------------------------------------

    def push_uci(self, uci: str) -> str:
        """Apply move; return SAN.  Raises ValueError if illegal."""
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {uci}")
        san = self.board.san(move)
        color = "white" if self.board.turn == chess.WHITE else "black"
        self.board.push(move)
        self._history.append({"uci": uci, "san": san, "color": color})
        return san

    def push(self, move: chess.Move) -> str:
        return self.push_uci(move.uci())

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def fen(self) -> str:
        return self.board.fen()

    def turn(self) -> str:
        return "white" if self.board.turn == chess.WHITE else "black"

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def outcome(self) -> Optional[str]:
        o = self.board.outcome()
        return o.result() if o else None

    def history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.board.reset()
        self._history.clear()

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
