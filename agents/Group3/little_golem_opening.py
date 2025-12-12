from __future__ import annotations

from typing import Optional, Tuple, List

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class LittleGolemOpening:
    """
    Little-Golem-style opening helper.

    Goals:
    - Provide a "neutral" first move for the first player (strong but not the single most swap-attractive).
    - Provide a swap decision for the second player based on how strong the opponent's first move is.
    - Never break the game: if anything is unexpected, return None and let MCTS/NN handle it.
    """

    def __init__(
        self,
        board_size: int = 11,
        neutral_band: int = 14,        # consider top-N moves by weight
        neutral_skip_top: int = 2,     # skip top-K (most swap-attractive) for first move
        swap_top_k: int = 5           # swap if opponent opening is among top-K strongest cells
    ):
        self.N = int(board_size)
        self.neutral_band = int(neutral_band)
        self.neutral_skip_top = int(neutral_skip_top)
        self.swap_top_k = int(swap_top_k)

        if self.N <= 0:
            raise ValueError("board_size must be positive")

        # Build a symmetric weight map (proxy for Little Golem opening preferences).
        self.weights = self._build_weight_map(self.N)

        # Pre-rank all coordinates once (fast at runtime).
        self.ranked_cells: List[Tuple[int, int]] = self._rank_cells()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def get_opening_move(
        self,
        board: Board,
        colour: Colour,
        opponent_move: Optional[Move]
    ) -> Optional[Move]:
        """
        Returns:
          - Move(x,y) for a neutral first move (when board is empty)
          - Move(-1,-1) to swap (when exactly one stone on board and it's advantageous)
          - None otherwise (let MCTS/NN decide)
        """
        try:
            total_stones = self._count_stones(board)
            if total_stones == 0:
                return self._neutral_first_move(board)

            if total_stones == 1 and opponent_move is not None:
                # Only meaningful for the second player's first decision.
                if self.should_swap(opponent_move, board):
                    return Move(-1, -1)
                return None

            return None
        except Exception:
            # Safety: never crash the agent because of opening logic.
            return None

    def should_swap(self, opponent_move: Move, board: Optional[Board] = None) -> bool:
        """
        Swap if opponent's opening is in the top-K strongest cells by weight.
        Optional 'board' lets us sanity-check legality, but isn't required.
        """
        try:
            x, y = int(opponent_move.x), int(opponent_move.y)
            if not self._in_bounds(x, y):
                return False

            # If board provided, ensure that cell is actually occupied (sanity).
            if board is not None:
                if getattr(board.tiles[x][y], "colour", None) is None:
                    return False

            top_moves = self.ranked_cells[: max(1, self.swap_top_k)]
            return (x, y) in top_moves
        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Weight map + ranking
    # ---------------------------------------------------------------------

    def _build_weight_map(self, N: int) -> List[List[float]]:
        """
        Symmetric centre-biased weight map.
        - Higher near centre = generally good opening influence.
        - Slight penalty on exact centre to reduce obvious swap-attractiveness.
        This is a simple proxy for a Little Golem opening preference table.
        """
        cx, cy = N // 2, N // 2
        W = [[0.0 for _ in range(N)] for _ in range(N)]

        for x in range(N):
            for y in range(N):
                # Smooth hill around centre (L1 distance).
                d = abs(x - cx) + abs(y - cy)
                base = 10.0 - 1.15 * d

                # Centre is typically very strong; we reduce it slightly for "neutrality".
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

    # ---------------------------------------------------------------------
    # Opening selection
    # ---------------------------------------------------------------------

    def _neutral_first_move(self, board: Board) -> Optional[Move]:
        """
        Neutral opening:
        - Look at top 'neutral_band' strong cells by weight,
        - Skip the very top 'neutral_skip_top' (most swap-attractive),
        - Choose first legal (empty) cell deterministically.
        """
        N = getattr(board, "size", self.N)
        if N != self.N:
            # If someone runs on a different board size, safest is to bail out.
            return None

        band = max(self.neutral_band, self.neutral_skip_top + 1)
        start = max(0, self.neutral_skip_top)
        candidates = self.ranked_cells[start:band]

        for x, y in candidates:
            if self._is_empty(board, x, y):
                return Move(x, y)

        # Fallback: best available anywhere
        for x, y in self.ranked_cells:
            if self._is_empty(board, x, y):
                return Move(x, y)

        return None

    # ---------------------------------------------------------------------
    # Utilities (defensive)
    # ---------------------------------------------------------------------

    def _count_stones(self, board: Board) -> int:
        N = getattr(board, "size", self.N)
        tiles = getattr(board, "tiles", None)
        if tiles is None:
            return 0

        cnt = 0
        for x in range(N):
            for y in range(N):
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
