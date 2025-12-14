from __future__ import annotations
from typing import Optional, Tuple, List

from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class LittleGolemOpening:
    """
    Stronger Little-Golem-style opening helper:
    - Neutral-but-strong first move (curated list)
    - Swap decision using a swap-zone + ranking
    - If not swapping: pattern reply (mirror, then local contest/bridge candidates)
    """

    def __init__(
        self,
        board_size: int = 11,
        swap_top_k: int = 5,
    ):
        self.N = int(board_size)
        self.swap_top_k = int(swap_top_k)

        if self.N <= 0:
            raise ValueError("board_size must be positive")

        self.cx, self.cy = self.N // 2, self.N // 2

        # Curated "neutral" openings: strong but not the most swap-attractive.
        # These are centred but avoid the exact centre + immediate obvious swap magnets.
        self.neutral_openings: List[Tuple[int, int]] = self._neutral_opening_list()

        # Weight map + ranked cells still used as fallback.
        self.weights = self._build_weight_map(self.N)
        self.ranked_cells: List[Tuple[int, int]] = self._rank_cells()

        # Explicit swap-zone (hot openings where swap is usually correct).
        self.swap_zone = self._build_swap_zone()

    # ---------------- Public API ----------------

    def get_opening_move(
        self,
        board: Board,
        colour: Colour,
        opponent_move: Optional[Move]
    ) -> Optional[Move]:
        """
        Returns:
          - Move(x,y) for a neutral first move (empty board)
          - Move(-1,-1) to swap (if opponent's first move is swap-attractive)
          - Move(x,y) pattern reply on second move if not swapping
          - None otherwise
        """
        try:
            stones = self._count_stones(board)

            # First move (empty board): choose neutral opening.
            if stones == 0:
                return self._neutral_first_move(board)

            # Second player's first decision (exactly 1 stone on board)
            if stones == 1 and opponent_move is not None:
                if self.should_swap(opponent_move, board):
                    return Move(-1, -1)
                # If not swapping, play a strong reply pattern.
                return self._second_move_reply(board, opponent_move)

            return None
        except Exception:
            return None

    def should_swap(self, opponent_move: Move, board: Optional[Board] = None) -> bool:
        """
        Stronger swap policy:
        - Swap if opponent move is in an explicit swap-zone (centre + ring-1 hot cells).
        - Otherwise fall back to "top-k by weight" (your old behaviour).
        """
        try:
            x, y = int(opponent_move.x), int(opponent_move.y)
            if not self._in_bounds(x, y):
                return False

            if board is not None:
                if getattr(board.tiles[x][y], "colour", None) is None:
                    return False

            # 1) swap-zone check (strong and simple)
            if (x, y) in self.swap_zone:
                return True

            # 2) top-k by weight (fallback)
            top_moves = self.ranked_cells[: max(1, self.swap_top_k)]
            return (x, y) in top_moves
        except Exception:
            return False

    # ---------------- Strong opening choices ----------------

    def _neutral_opening_list(self) -> List[Tuple[int, int]]:
        """
        Curated neutral openings around centre.
        Avoids exact centre and the most swap-attractive immediate neighbours.
        Deterministic order: early choices are best.
        """
        c = (self.cx, self.cy)

        # Ring-1 around centre (hex-grid-ish feel; we approximate with square coords)
        ring1 = [
            (c[0]-1, c[1]), (c[0]+1, c[1]),
            (c[0], c[1]-1), (c[0], c[1]+1),
            (c[0]-1, c[1]-1), (c[0]+1, c[1]+1),
        ]

        # Ring-2 “still central but calmer”
        ring2 = [
            (c[0]-2, c[1]), (c[0]+2, c[1]),
            (c[0], c[1]-2), (c[0], c[1]+2),
            (c[0]-2, c[1]-2), (c[0]+2, c[1]+2),
            (c[0]-2, c[1]-1), (c[0]-1, c[1]-2),
            (c[0]+2, c[1]+1), (c[0]+1, c[1]+2),
        ]

        # We *exclude* exact centre on purpose (too swap-attractive).
        candidates = ring2 + ring1  # prefer slightly calmer than immediate ring-1
        return [p for p in candidates if self._in_bounds(p[0], p[1])]

    def _neutral_first_move(self, board: Board) -> Optional[Move]:
        # Try curated list first
        for x, y in self.neutral_openings:
            if self._is_empty(board, x, y):
                return Move(x, y)

        # Fallback to best by weight
        for x, y in self.ranked_cells:
            if self._is_empty(board, x, y):
                return Move(x, y)
        return None

    def _second_move_reply(self, board: Board, opponent_move: Move) -> Optional[Move]:
        """
        Pattern reply when you decide NOT to swap:
        1) Mirror move across centre (common strong reply)
        2) Otherwise play a local contest/bridge candidate near opponent move
        3) Otherwise fallback to weighted best
        """
        ox, oy = int(opponent_move.x), int(opponent_move.y)

        # 1) mirror across centre
        mx, my = (self.N - 1 - ox, self.N - 1 - oy)
        if self._in_bounds(mx, my) and self._is_empty(board, mx, my):
            return Move(mx, my)

        # 2) local contest / "bridge-ish" candidates near opponent
        local = [
            (ox-1, oy), (ox+1, oy),
            (ox, oy-1), (ox, oy+1),
            (ox-1, oy-1), (ox+1, oy+1),

            # slightly further (template-ish)
            (ox-2, oy), (ox+2, oy),
            (ox, oy-2), (ox, oy+2),
            (ox-2, oy-1), (ox-1, oy-2),
            (ox+2, oy+1), (ox+1, oy+2),
        ]
        for x, y in local:
            if self._in_bounds(x, y) and self._is_empty(board, x, y):
                return Move(x, y)

        # 3) fallback to weighted best
        for x, y in self.ranked_cells:
            if self._is_empty(board, x, y):
                return Move(x, y)

        return None

    # ---------------- Swap-zone + weights ----------------

    def _build_swap_zone(self) -> set[Tuple[int, int]]:
        """
        Swap-zone = centre + immediate ring1 + a few extra hot cells.
        This is intentionally aggressive because failing to swap loses games.
        """
        c = (self.cx, self.cy)
        ring1 = {
            (c[0], c[1]),
            (c[0]-1, c[1]), (c[0]+1, c[1]),
            (c[0], c[1]-1), (c[0], c[1]+1),
            (c[0]-1, c[1]-1), (c[0]+1, c[1]+1),
        }

        # Add a couple of “almost-centre” cells that are still swap-attractive
        extra = {
            (c[0]-2, c[1]-1), (c[0]-1, c[1]-2),
            (c[0]+2, c[1]+1), (c[0]+1, c[1]+2),
        }

        zone = set()
        for p in ring1 | extra:
            if self._in_bounds(p[0], p[1]):
                zone.add(p)
        return zone

    def _build_weight_map(self, N: int) -> List[List[float]]:
        """
        Centre-biased map with stronger penalties for being 'too central' (swap magnets).
        Used as fallback ranking, not primary opening choice.
        """
        cx, cy = self.cx, self.cy
        W = [[0.0 for _ in range(N)] for _ in range(N)]

        for x in range(N):
            for y in range(N):
                d = abs(x - cx) + abs(y - cy)
                base = 10.0 - 1.10 * d

                # penalise centre + ring-1 (swap magnets)
                if (x, y) == (cx, cy):
                    base -= 2.0
                if abs(x - cx) + abs(y - cy) == 1:
                    base -= 0.8

                if base < 0.0:
                    base = 0.0
                W[x][y] = base

        return W

    def _rank_cells(self) -> List[Tuple[int, int]]:
        cells = [(x, y) for x in range(self.N) for y in range(self.N)]
        cells.sort(key=lambda p: self.weights[p[0]][p[1]], reverse=True)
        return cells

    # ---------------- Utilities ----------------

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
