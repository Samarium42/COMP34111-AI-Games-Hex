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


class NumbaGraveNN(AgentBase):
    def __init__(self, colour: Colour,
                 load_path="models/hex11-20180712-3362.policy.pth",
                 sims=800,
                 c_puct=1.2):

        super().__init__(colour)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[NumbaGraveNN] Loading Azalea network")
        self.net = load_hex11_pretrained(load_path, self.device)
        self.net.eval()

        self.sims = sims

        # Single C++ engine instance
        # Assumes CppMCTS(board_size, sims, c_puct) or adjust as needed
        self.tree = CppMCTS(board_size=11, sims=sims)

    def opening_move(self, turn, board: Board, opponent_move: Move | None):
        N = board.size
        centre = (N // 2, N // 2)
        near = {
           (centre[0] - 1, centre[1]),
           (centre[0] + 1, centre[1]),
           (centre[0], centre[1] - 1),
           (centre[0], centre[1] + 1),
           }

        if turn == 1 and self.colour == Colour.RED:
           return Move(centre[0] - 1, centre[1] - 1)  # (4,4)

        if turn == 2 and self.colour == Colour.BLUE:
           if opponent_move is None:
               return None

           ox, oy = opponent_move.x, opponent_move.y

           if (ox, oy) == centre or (ox, oy) in near:
               return Move.swap()
           return None

        return None    

    @staticmethod
    def canonicalise_board(leaf_board: np.ndarray, leaf_player: int, N: int) -> np.ndarray:
        b = leaf_board.copy()
        if leaf_player == 2:  # BLUE to move
            blue_mask = (b == 2)
            red_mask = (b == 1)

            b[blue_mask] = 1
            b[red_mask] = 2
        return b.reshape(N, N)    

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
                cannonical_board = self.canonicalise_board(leaf_board, leaf_player, N)
                # non-terminal: call NN
                encoded = torch.tensor(
                    cannonical_board,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                logits, value_t = self.net(encoded)
                value = float(value_t.item())

                raw = torch.softmax(logits[0], dim=0).cpu().numpy()

                # mask to legal moves
                priors = np.zeros(N * N, dtype=np.float64)
                empties = np.where(leaf_board == 0)[0]
                if len(empties) > 0:
                    priors[empties] = raw[empties]
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
            "[NumbaGraveNN] Move =", action,
            "took", time.time() - t0, "seconds"
            f"(NN calls: {nn_calls}, terminal hits: {terminal_hits})"
            "sims =", self.sims,
        )

        return Move(action // N, action % N)
