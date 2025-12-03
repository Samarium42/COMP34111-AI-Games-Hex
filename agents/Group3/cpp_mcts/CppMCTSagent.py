# agents/Group3/CppMCTSAgent.py

import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour

from agents.Group3.cpp_mcts import best_move


class CppMCTSAgent(AgentBase):
    def __init__(self, colour: Colour,
                 sims: int = 500,
                 c_puct: float = 1.4):
        super().__init__(colour)
        self.sims = sims
        self.c_puct = c_puct

    def _board_to_flat(self, board: Board):
        N = board.size
        flat = np.zeros(N * N, dtype=np.int32)
        for x in range(N):
            for y in range(N):
                c = board.tiles[x][y].colour
                idx = x * N + y
                if c == Colour.RED:
                    flat[idx] = 1
                elif c == Colour.BLUE:
                    flat[idx] = 2
                else:
                    flat[idx] = 0
        return flat

    def make_move(self, colour: Colour, board: Board, opponent_move: Move | None) -> Move:
        N = board.size
        board_flat = self._board_to_flat(board)
        player_int = 1 if colour == Colour.RED else 2

        move_idx = best_move(board_flat, N, player_int,
                             sims=self.sims, c_puct=self.c_puct)

        x = move_idx // N
        y = move_idx % N
        return Move(x, y)
