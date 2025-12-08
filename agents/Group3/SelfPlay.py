# agents/Group3/SelfPlayGraveNN.py

import time
import numpy as np
import torch

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour

from agents.Group3.azalea_net import load_hex11_pretrained
from agents.Group3.cpp_mcts.interface import CppMCTS


class HexState:
    __slots__ = ["N", "player", "board_flat"]

    def __init__(self, board: Board, player: Colour):
        """
        board_flat: shape (N*N,), 0=empty, 1=RED, 2=BLUE
        player: 1 if RED to move, 2 if BLUE to move
        """
        self.N = board.size
        self.player = 1 if player == Colour.RED else 2

        arr = np.zeros(self.N * self.N, dtype=np.int32)
        for x in range(self.N):
            for y in range(self.N):
                c = board.tiles[x][y].colour
                if c == Colour.RED:
                    arr[x * self.N + y] = 1
                elif c == Colour.BLUE:
                    arr[x * self.N + y] = 2
        self.board_flat = arr


# ======================================================================
# Self-play NN + C++ MCTS agent
# ======================================================================

class SelfPlayGraveNN(AgentBase):
    """
    MCTS + neural net agent that uses ONLY the self play weights.

    Use in Hex.py as:
        -p1 "agents.Group3.SelfPlayGraveNN SelfPlayGraveNN"
    """

    def __init__(
        self,
        colour: Colour,
        load_path: str = "agents/Group3/models/selfplay_iter_10.pth",
        sims: int = 2000,
        c_puct: float = 1.2,
    ):
        self.is_learning_agent = False

        super().__init__(colour)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SelfPlayGraveNN] Loading self play network from {load_path}")
        self.net = load_hex11_pretrained(load_path, self.device)
        self.net.eval()

        self.sims = sims
        self.tree = CppMCTS(board_size=11, sims=sims, c_puct=c_puct)

    def opening_move(self, turn: int, board: Board, opponent_move: Move | None):
        N = board.size
        centre = (N // 2, N // 2)
        near = {
            (centre[0] - 1, centre[1]),
            (centre[0] + 1, centre[1]),
            (centre[0], centre[1] - 1),
            (centre[0], centre[1] + 1),
            (centre[0] - 1, centre[0] - 1),
            (centre[0] + 1, centre[0] + 1),
        }

        # First move as Red: play near centre
        if turn == 1 and self.colour == Colour.RED:
            return Move(centre[0] - 1, centre[1] - 1)

        # Turn 2 as Blue: decide whether to swap
        if turn == 2 and self.colour == Colour.BLUE:
            if opponent_move is None:
                return None
            ox, oy = opponent_move.x, opponent_move.y
            if (ox, oy) == centre or (ox, oy) in near:
                # swap if they got a strong centre stone
                return Move(-1, -1)
            return None

        return None

    @staticmethod
    def canonicalise_board(leaf_board: np.ndarray, leaf_player: int, N: int):
        """
        Convert a (flat) Hex board to a canonical orientation for the net.

        Input:
            leaf_board: np.ndarray shape (N*N,), 0=empty, 1=RED, 2=BLUE
            leaf_player: 1 or 2 (player to move at this node)
            N: board size

        Returns:
            canon_board: np.ndarray shape (N, N) int32 for the NN
            idx_map: np.ndarray shape (N*N,), so that

                raw_unmapped[orig_idx] = raw_canon[idx_map[orig_idx]]

            where raw_canon are logits in canonical orientation.
        """
        b = leaf_board.reshape(N, N).copy()

        if leaf_player == 1:
            # Red to move (top-bottom). Already matches the training side.
            canon = b
            idx_map = np.arange(N * N, dtype=np.int32)
        else:
            # Blue to move (left-right).
            # 1) swap colours so Blue -> 1 (current player), Red -> 2 (opponent)
            swapped = b.copy()
            blue_mask = swapped == 2
            red_mask = swapped == 1
            swapped[blue_mask] = 1
            swapped[red_mask] = 2

            # 2) transpose so left-right in original becomes top-bottom here
            canon = swapped.T  # shape (N, N)

            # Build index map: original flat index -> canonical flat index.
            # orig idx = x*N + y
            # after transpose: new_x = y, new_y = x => canon idx = y*N + x
            idx_map = np.arange(N * N, dtype=np.int32).reshape(N, N).T.reshape(-1)

        return canon, idx_map

    # ------------------------------------------------------------------
    # Main move function with C++ MCTS + self-play net
    # ------------------------------------------------------------------
    @torch.no_grad()
    def make_move(self, turn: int, board: Board, opponent_move: Move | None) -> Move:
        # Encode current position
        state = HexState(board, self.colour)
        root_board = state.board_flat
        root_player = state.player
        N = state.N

        # Initialise C++ tree at root
        self.tree.reset(root_board, root_player)

        # Opening heuristic
        opening = self.opening_move(turn, board, opponent_move)
        if opening is not None:
            return opening

        t0 = time.time()
        nn_calls = 0
        terminal_hits = 0

        # MCTS loop
        for _ in range(self.sims):
            # Ask C++ engine for a leaf
            leaf_board, leaf_player, is_terminal = self.tree.request_leaf()
            leaf_board = np.asarray(leaf_board, dtype=np.int32)

            if is_terminal == 1:
                # Terminal node: uniform priors on legal moves, value 0
                terminal_hits += 1

                empties = np.where(leaf_board == 0)[0]
                priors = np.zeros(N * N, dtype=np.float64)
                if len(empties) > 0:
                    priors[empties] = 1.0 / len(empties)
                value = 0.0
            else:
                # Non terminal: evaluate with self play net
                nn_calls += 1

                canon_board, idx_map = SelfPlayGraveNN.canonicalise_board(
                    leaf_board, leaf_player, N
                )

                encoded = torch.tensor(
                    canon_board,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)  # (1, N, N)

                logits, value_t = self.net(encoded)  # logits: (1, N*N), value_t: (1,1) or (1,)
                value = float(value_t.view(-1)[0].item())

                raw_canon = torch.softmax(logits[0], dim=0).cpu().numpy()  # (N*N,)

                # Map priors back to original orientation
                raw_unmapped = np.zeros_like(raw_canon)
                raw_unmapped[:] = raw_canon[idx_map]

                priors = np.zeros(N * N, dtype=np.float64)
                empties = np.where(leaf_board == 0)[0]
                if len(empties) > 0:
                    priors[empties] = raw_unmapped[empties]
                    s = priors.sum()
                    if s > 0:
                        priors /= s
                    else:
                        priors[empties] = 1.0 / len(empties)
                else:
                    priors[:] = 1.0 / (N * N)

            # Feed eval back to C++ side
            self.tree.apply_eval(priors, value)

        # Pick best root action by visit count
        action = int(self.tree.best_action())

        print(
            "[SelfPlayGraveNN] Move =",
            action,
            "took",
            time.time() - t0,
            "seconds",
            f"(NN calls: {nn_calls}, terminal hits: {terminal_hits})",
            "sims =",
            self.sims,
        )

        return Move(action // N, action % N)
