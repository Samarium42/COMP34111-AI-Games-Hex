import torch
import numpy as np
import time 

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour

from agents.Group3.azalea_net import load_hex11_pretrained
from agents.Group3.cpp_mcts.interface import CppMCTS


class HexState:
    __slots__ = ["N", "player", "board_flat"]

    def __init__(self, board: Board, player: Colour):
        self.N = board.size
        # 1 = RED, 2 = BLUE (match what C++ expects)
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

    def encode(self, device):
        """
        Encode for Azalea: (1, N, N) long
        """
        return torch.tensor(
            self.board_flat.reshape(self.N, self.N),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)


class AzaleaGraveNN(AgentBase):
    def __init__(self, colour: Colour,
                 load_path="agents/Group3/models/hex11-20180712-3362.policy.pth",
                 sims=2000,
                 c_puct = 1.2,
                 use_grave = True,
                 grave_ref = 0.5):

        super().__init__(colour)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_hex11_pretrained(load_path, self.device)
        self.net.eval()

        self.sims = sims

        # Single C++ engine instance
        # Assumes CppMCTS(board_size, sims, c_puct) or adjust as needed
        self.tree = CppMCTS(board_size=11, 
                            sims=sims,
                            c_puct=c_puct,     
                            use_grave=use_grave, 
                            grave_ref=grave_ref)  

    def opening_move(self, turn, board: Board, opponent_move: Move | None):
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

        if turn == 1 and self.colour == Colour.RED:
           return Move(centre[0] - 1, centre[1] - 1)  # (4,4)

        if turn == 2 and self.colour == Colour.BLUE:
           if opponent_move is None:
               return None

           ox, oy = opponent_move.x, opponent_move.y

           if (ox, oy) == centre or (ox, oy) in near:
               return Move(-1,-1)
           return None

        return None    

    @staticmethod
    def canonicalise_board(leaf_board: np.ndarray, leaf_player: int, N: int):
        """
        Return:
            canon_board: (N, N) int32 for the NN
            idx_map: np.ndarray shape (N*N,) such that

                raw_unmapped[idx_orig] = raw_canon[idx_map[idx_orig]]

            i.e. idx_map maps ORIGINAL flat index -> CANONICAL flat index.
        """
        b = leaf_board.reshape(N, N).copy()

        if leaf_player == 1:
            # Red to move, connects top-bottom already.
            # Just ensure '1' = player, '2' = opp (it already is).
            canon = b
            idx_map = np.arange(N * N, dtype=np.int32)
        else:
            # Blue to move, connects left-right.
            # 1) swap colours so BLUE -> 1 (player), RED -> 2 (opponent)
            swapped = b.copy()
            blue_mask = swapped == 2
            red_mask = swapped == 1
            swapped[blue_mask] = 1
            swapped[red_mask] = 2

            # 2) transpose so left-right in original becomes top-bottom here
            canon = swapped.T  # shape (N, N)

            # 3) build index map: original idx -> canonical idx
            # original idx = x*N + y  (x=row, y=col)
            # after transpose: new_x = y, new_y = x
            # canonical idx = new_x*N + new_y = y*N + x
            idx_map = np.arange(N * N, dtype=np.int32).reshape(N, N).T.reshape(-1)

        return canon, idx_map
  

    @torch.no_grad()
    def make_move(self, turn, board: Board, opponent_move: Move | None):

        state = HexState(board, self.colour)
        root_board = state.board_flat        # np.ndarray shape (N*N,)
        root_player = state.player           # 1 or 2
        N = state.N

        # initialise C++ tree at root
        self.tree.reset(root_board, root_player)

        t0 = time.time()
        nn_calls = 0
        terminal_hits = 0
        opening = self.opening_move(turn, board, opponent_move)
        if opening is not None:
            return opening

        # run MCTS in C++ with NN in Python
        for _ in range(self.sims):
            # ask C++ engine for a leaf to evaluate
            leaf_board, leaf_player, is_terminal = self.tree.request_leaf()
            leaf_board = np.asarray(leaf_board, dtype=np.int32)

            # compute priors and value
            if is_terminal == 1:
                terminal_hits += 1
                # terminal: uniform over legal moves as dummy prior, zero value
                empties = np.where(leaf_board == 0)[0]
                priors = np.zeros(N * N, dtype=np.float64)
                if len(empties) > 0:
                    priors[empties] = 1.0 / len(empties)
                value = 0.0
            else:
                nn_calls += 1

                canon_board, idx_map = self.canonicalise_board(
                    leaf_board, leaf_player, N
                )

                # non-terminal: call NN on canonical board
                encoded = torch.tensor(
                    canon_board,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)  # (1, N, N)

                logits, value_t = self.net(encoded)  # logits: (1, N*N)
                value = float(value_t.item())

                raw_canon = torch.softmax(logits[0], dim=0).cpu().numpy()  # (N*N,)

                # map back to original orientation
                raw_unmapped = np.zeros_like(raw_canon)
                # idx_map[orig_idx] = canon_idx
                raw_unmapped[:] = raw_canon[idx_map]

                # mask to legal moves on ORIGINAL board
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

            # feed eval back to C++ side
            self.tree.apply_eval(priors, value)

        # after all sims, ask engine for best action at root
        action = int(self.tree.best_action())

        print(
            "[CPPGraveNN] Move =", action,
            "took", time.time() - t0, "seconds"
            f"(NN calls: {nn_calls}, terminal hits: {terminal_hits})"
            "sims =", self.sims,
        )

        return Move(action // N, action % N)
