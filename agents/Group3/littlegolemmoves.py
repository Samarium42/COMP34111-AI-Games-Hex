# agents/Group3/littlegolemmoves.py
from __future__ import annotations

from typing import Optional, List, Tuple

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class LittleGolemOpening:
    """
    Little-Golem-style opening helper (defensive + deterministic).

    What it does:
    1) If the board is empty: play a strong-but-neutral first move (not the absolute best swap-bait).
    2) If there is exactly one stone and we are the second player: decide whether to swap.
    3) Otherwise: return None and let your MCTS/NN decide.

    It NEVER throws. If anything is unexpected, it returns None.
    """

    def _init_(
        self,
        board_size: int = 11,
        neutral_band: int = 14,        # consider top-N moves by weight for first move
        neutral_skip_top: int = 2,     # skip top-K strongest (swap-attractive) moves
        swap_top_k: int = 5            # swap if opponent first move is among top-K strongest
    ):
        self.N = int(board_size)
        self.neutral_band = int(neutral_band)
        self.neutral_skip_top = int(neutral_skip_top)
        self.swap_top_k = int(swap_top_k)

        # Precompute ranking once.
        self.weights = self._build_weight_map(self.N)
        self.ranked_cells: List[Tuple[int, int]] = self._rank_cells()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def get_opening_move(
        self,
        board: Board,
        colour: Colour,
        opponent_move: Optional[Move]
    ) -> Optional[Move]:
        """
        Returns:
          - Move(x,y) for a neutral first move (board empty)
          - Move(-1,-1) to swap (board has exactly one stone and swap is favourable)
          - None otherwise (let MCTS/NN decide)
        """
        try:
            # Safety: board size mismatch => bail
            if getattr(board, "size", self.N) != self.N:
                return None

            total_stones = self._count_stones(board)

            # Board empty: choose a neutral strong first move
            if total_stones == 0:
                return self._neutral_first_move(board)

            # One stone: we are responding to opponent opening; only meaningful if opponent_move is provided
            if total_stones == 1 and opponent_move is not None:
                # Swap is only relevant for the second player's first decision.
                # We don't need "turn" — stone count is reliable.
                if self.should_swap(opponent_move, board):
                    return Move(-1, -1)
                return None

            return None
        except Exception:
            return None

    def should_swap(self, opponent_move: Move, board: Optional[Board] = None) -> bool:
        """
        Swap if opponent opening is among top-K strongest cells by our symmetric weight map.
        """
        try:
            ox, oy = int(opponent_move.x), int(opponent_move.y)
            if not self._in_bounds(ox, oy):
                return False

            # Optional sanity check: if board provided, that cell should be occupied
            if board is not None:
                if getattr(board.tiles[ox][oy], "colour", None) is None:
                    return False

            k = max(1, self.swap_top_k)
            top_moves = self.ranked_cells[:k]
            return (ox, oy) in top_moves
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Opening policies
    # ------------------------------------------------------------------

    def _neutral_first_move(self, board: Board) -> Optional[Move]:
        """
        Deterministic neutral opening:
        - Consider the top 'neutral_band' strong cells
        - Skip the very top 'neutral_skip_top' cells (most swap-attractive)
        - Choose the first legal empty candidate
        """
        try:
            band = max(self.neutral_band, self.neutral_skip_top + 1)
            start = max(0, self.neutral_skip_top)
            candidates = self.ranked_cells[start:band]

            for x, y in candidates:
                if self._is_empty(board, x, y):
                    return Move(x, y)

            # Fallback: best empty anywhere
            for x, y in self.ranked_cells:
                if self._is_empty(board, x, y):
                    return Move(x, y)

            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Weight map (proxy for Little Golem opening preferences)
    # ------------------------------------------------------------------

    def _build_weight_map(self, N: int) -> List[List[float]]:
        """
        Symmetric centre-biased weight map.

        Intuition:
        - Centre-ish moves are generally strong openers in Hex.
        - But the absolute centre is often too strong and encourages swapping.
          We slightly reduce exact centre to bias towards a "neutral" first move.
        """
        cx, cy = N // 2, N // 2
        W = [[0.0 for _ in range(N)] for _ in range(N)]

        for x in range(N):
            for y in range(N):
                # L1 distance from centre
                d = abs(x - cx) + abs(y - cy)

                # A smooth hill: high in centre, decreasing outwards
                base = 10.0 - 1.15 * d

                # Slight penalty on exact centre to reduce swap-bait
                if (x, y) == (cx, cy):
                    base -= 0.9

                if base < 0.0:
                    base = 0.0

                W[x][y] = base

        return W

    def _rank_cells(self) -> List[Tuple[int, int]]:
        cells = [(x, y) for x in range(self.N) for y in range(self.N)]
        cells.sort(key=lambda p: self.weights[p[0]][p[1]], reverse=True)
        return cells

    # ------------------------------------------------------------------
    # Utilities (defensive)
    # ------------------------------------------------------------------

    def _count_stones(self, board: Board) -> int:
        tiles = getattr(board, "tiles", None)
        if tiles is None:
            return 0

        cnt = 0
        for x in range(self.N):
            for y in range(self.N):
                if getattr(tiles[x][y], "colour", None) is not None:
                    cnt += 1
        return cnt

    def _is_empty(self, board: Board, x: int, y: int) -> bool:
        try:
            return getattr(board.tiles[x][y], "colour", None) is None
        except Exception:
            return False

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.N and 0 <= y < self.N