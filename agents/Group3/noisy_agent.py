# agents/Group3/noisy_agent.py

import numpy as np
from agents.Group3.NumbaGraveNN import NumbaGraveNN
from src.Move import Move

class NoisyNumbaGraveNN(NumbaGraveNN):
    """
    Same net and MCTS, but at the root we sample move
    with some temperature instead of always taking argmax.
    Very crude but good enough as a noisy opponent.
    """

    def make_move(self, turn, board, opponent_move):
        # Call parent to run full MCTS
        move = super().make_move(turn, board, opponent_move)

        # Optional: with small probability, override with semi-random move
        # based on NN priors (no extra MCTS needed)
        eps = 0.15  # 15% of the time, explore
        if np.random.rand() > eps:
            return move

        # Quick fallback: choose a random legal move
        N = board.size
        empties = []
        for x in range(N):
            for y in range(N):
                if board.tiles[x][y].colour is None:
                    empties.append((x, y))

        if not empties:
            return move

        rx, ry = empties[np.random.randint(len(empties))]
        return Move(rx, ry)
