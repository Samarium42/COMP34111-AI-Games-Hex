# ======================
#  grave_agent.py
# ======================

import numpy as np
import torch
import torch.nn.functional as F

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

from mcts_core import HexState, MCTS  # ⬅ COMPILED CYTHON VERSION
from GraveNN import HexResNet        # your NN stays Python / Torch


class GraveNNAgent(AgentBase):

    def __init__(self, sims=200, board_size=11, device="cpu"):
        super().__init__()

        self.board_size = board_size
        self.net = HexResNet(board_size)
        self.net.eval()
        self.device = device

        self.mcts = MCTS(self.net, sims=sims, c_puct=1.2, device=device)


    def get_move(self, board: Board, colour: Colour, move=None):
        state = HexState(board.copy(), colour)
        counts = self.mcts.run(state)

        best_action = int(np.argmax(counts))
        if best_action == self.board_size*self.board_size:  # swap move
            return Move(-1, -1)

        x = best_action // self.board_size
        y = best_action % self.board_size
        return Move(x, y)
