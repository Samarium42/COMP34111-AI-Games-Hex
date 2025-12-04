"""
A minimal, original opening book for 11x11 Hex.

This module does not use known openings or external data.
It uses only general principles:

1. Centre is most flexible, but invites swap.
2. Slightly off-centre is still strong but less swappable.
3. Early game should concentrate stones in a compact cluster.
4. Responses should prefer symmetry and maintain central influence.

This yields a simple, safe, non-derivative opening guide.
"""

from typing import Optional, Tuple
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class OpeningBook:
    def __init__(self, board_size: int = 11):
        self.N = board_size
        self.centre = (board_size // 2, board_size // 2)

        # A ring around the centre (distance 1)
        cx, cy = self.centre
        self.near_centre = [
            (cx - 1, cy), (cx + 1, cy),
            (cx, cy - 1), (cx, cy + 1),
            (cx - 1, cy - 1), (cx + 1, cy + 1),
        ]

    def get_move(self, board: Board, turn: int, colour: Colour) -> Optional[Move]:
        if turn == 1:
            return self._first_player_open()

        if turn == 3:
            return self._third_move_reply(board, colour)

        if turn <= 5:
            return self._early_compact_cluster(board)

        return None

    def _first_player_open(self) -> Move:
        """
        Play slightly off-centre to avoid an obvious swap.
        Pure centre is strongest but fully symmetric.
        """
        # Choose the first available near-centre cell
        for x, y in self.near_centre:
            if 0 <= x < self.N and 0 <= y < self.N:
                return Move(x, y)

        # Fallback to centre
        cx, cy = self.centre
        return Move(cx, cy)

    def _third_move_reply(self, board: Board, colour: Colour) -> Optional[Move]:
        """
        After opponent did not swap, aim for a compact cluster.
        """
        stones = self._list_stones(board, exclude_colour=colour)
        if not stones:
            return None

        ox, oy = stones[0]
        cx, cy = self.centre

        # If opponent claimed centre, take a nearby cell
        if (ox, oy) == self.centre:
            for cell in self.near_centre:
                x, y = cell
                if board.tiles[x][y].colour is None:
                    return Move(x, y)

        # Otherwise, claim centre if free
        if board.tiles[cx][cy].colour is None:
            return Move(cx, cy)

        return None

    def _early_compact_cluster(self, board: Board) -> Optional[Move]:
        """
        Try to place stones within distance <= 2 of the centre.
        """
        cx, cy = self.centre
        for x in range(cx - 2, cx + 3):
            for y in range(cy - 2, cy + 3):
                if 0 <= x < self.N and 0 <= y < self.N:
                    if board.tiles[x][y].colour is None:
                        return Move(x, y)
        return None

    def should_swap(self, opponent_opening: Tuple[int, int]) -> bool:
        """
        Very simple symmetric rule:
        swap only if opponent played exactly in the centre.
        """
        return opponent_opening == self.centre

    def _list_stones(self, board: Board, exclude_colour: Colour):
        N = board.size
        out = []
        for i in range(N):
            for j in range(N):
                c = board.tiles[i][j].colour
                if c is not None and c != exclude_colour:
                    out.append((i, j))
        return out
