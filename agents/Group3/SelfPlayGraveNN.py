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
from agents.Group3.littlegolemmoves import LittleGolemOpening



class HexState:
    __slots__ = ["N", "player", "board_flat"]

    def __init__(self, board: Board, player: Colour):
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


class SelfPlayGraveNN(AgentBase):
    def __init__(
        self,
        colour: Colour,
        load_path: str = "agents/Group3/models/selfplay_iter_10.pth",
        sims: int = 3000,
        c_puct: float = 1.2,
        use_grave: bool = True,
        grave_ref: float = 0.5,
        batch_size: int = 32,
    ):
        self.is_learning_agent = False
        super().__init__(colour)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SelfPlayGraveNN] Loading self play network from {load_path}")
        self.net = load_hex11_pretrained(load_path, self.device)
        self.net.eval()

        self.sims = int(sims)
        self.batch_size = int(batch_size)

        try:
            self.tree = CppMCTS(
                board_size=11,
                sims=self.sims,
                c_puct=c_puct,
                use_grave=use_grave,
                grave_ref=grave_ref,
                batch_size=self.batch_size,
            )
        except TypeError:
            self.tree = CppMCTS(board_size=11, sims=self.sims, c_puct=c_puct)
            try:
                self.tree.set_grave_enabled(use_grave)
                self.tree.set_grave_ref(grave_ref)
            except Exception:
                pass

    def opening_move(self, turn: int, board: Board, opponent_move: Move | None):
        if not hasattr(self, "lg_opening"):
            self.lg_opening = LittleGolemOpening(board_size=board.size)

        return self.lg_opening.get_opening_move(board, self.colour, opponent_move)


    @staticmethod
    def _uniform_priors(leaf_board: np.ndarray, N: int) -> np.ndarray:
        priors = np.zeros(N * N, dtype=np.float64)
        empties = np.where(leaf_board == 0)[0]
        if empties.size > 0:
            priors[empties] = 1.0 / float(empties.size)
        else:
            priors[:] = 1.0 / float(N * N)
        return priors

    @staticmethod
    def canonicalise_board(leaf_board: np.ndarray, leaf_player: int, N: int):
        """
        Canonicalise so the NN always sees "player to move" as 1 and plays top to bottom.

        Input:
            leaf_board: flat (N*N,) with 0 empty, 1 red, 2 blue (original orientation)
            leaf_player: 1 or 2 (player to move in original state)

        Returns:
            canon_board: (N, N) int32 board to feed NN
            idx_map: (N*N,) int32 mapping ORIGINAL flat index -> CANONICAL flat index
                     so raw_unmapped[orig_idx] = raw_canon[idx_map[orig_idx]]
        """
        b = leaf_board.reshape(N, N)

        if leaf_player == 1:
            canon = b.copy()
            idx_map = np.arange(N * N, dtype=np.int32)
            return canon, idx_map

        swapped = b.copy()
        blue_mask = swapped == 2
        red_mask = swapped == 1
        swapped[blue_mask] = 1
        swapped[red_mask] = 2

        canon = swapped.T

        idx_map = np.arange(N * N, dtype=np.int32).reshape(N, N).T.reshape(-1)
        return canon, idx_map
    


    @torch.no_grad()
    def make_move(self, turn: int, board: Board, opponent_move: Move | None) -> Move:
        opening = self.opening_move(turn, board, opponent_move)
        if opening is not None:
            return opening
        state = HexState(board, self.colour)
        root_board = state.board_flat
        root_player = state.player
        N = state.N
        NN = N * N

        opening = self.opening_move(turn, board, opponent_move)
        if opening is not None:
            return opening

        self.tree.reset(root_board, root_player)

        t0 = time.time()
        nn_calls = 0
        terminal_hits = 0

        sims_done = 0
        BMAX = max(1, int(self.batch_size))

        while sims_done < self.sims:
            B = min(BMAX, self.sims - sims_done)

            if not hasattr(self.tree, "request_leaves") or not hasattr(self.tree, "apply_evals_batch"):
                for _ in range(B):
                    leaf_board, leaf_player, is_terminal = self.tree.request_leaf()
                    leaf_board = np.asarray(leaf_board, dtype=np.int32)

                    if is_terminal == 1:
                        terminal_hits += 1
                        priors = SelfPlayGraveNN._uniform_priors(leaf_board, N)
                        value = 0.0
                    else:
                        canon_board, idx_map = SelfPlayGraveNN.canonicalise_board(
                            leaf_board, int(leaf_player), N
                        )

                        encoded = torch.as_tensor(
                            canon_board,
                            dtype=torch.long,
                            device=self.device,
                        ).unsqueeze(0)

                        policy_logits, value_t = self.net(encoded)
                        value = float(value_t.view(-1)[0].item())
                        nn_calls += 1

                        raw_canon = torch.softmax(policy_logits[0], dim=0).detach().cpu().numpy()
                        raw_unmapped = raw_canon[idx_map]

                        priors = np.zeros(NN, dtype=np.float64)
                        empties = np.where(leaf_board == 0)[0]
                        if empties.size > 0:
                            priors[empties] = raw_unmapped[empties]
                            s = float(priors.sum())
                            if s > 0.0:
                                priors /= s
                            else:
                                priors[empties] = 1.0 / float(empties.size)
                        else:
                            priors[:] = 1.0 / float(NN)

                    self.tree.apply_eval(priors, value)

                sims_done += B
                continue

            boards_batch, players_batch, terms_batch = self.tree.request_leaves(B)

            priors_batch = np.zeros((B, NN), dtype=np.float64)
            values_batch = np.zeros((B,), dtype=np.float64)

            term_idx = np.where(terms_batch == 1)[0]
            if term_idx.size > 0:
                terminal_hits += int(term_idx.size)
                for i in term_idx:
                    leaf_board = boards_batch[i].astype(np.int32, copy=False)
                    priors_batch[i, :] = SelfPlayGraveNN._uniform_priors(leaf_board, N)
                    values_batch[i] = 0.0

            non_idx = np.where(terms_batch == 0)[0]
            if non_idx.size > 0:
                non_boards = boards_batch[non_idx].astype(np.int32, copy=False)
                non_players = players_batch[non_idx].astype(np.int32, copy=False)

                key = np.concatenate([non_players.reshape(-1, 1), non_boards], axis=1)
                uniq_key, inv = np.unique(key, axis=0, return_inverse=True)

                uniq_players = uniq_key[:, 0].astype(np.int32, copy=False)
                uniq_boards = uniq_key[:, 1:].astype(np.int32, copy=False)

                U = int(uniq_boards.shape[0])
                nn_calls += U

                canon_stack = np.empty((U, N, N), dtype=np.int32)
                idx_maps = np.empty((U, NN), dtype=np.int32)

                for u in range(U):
                    canon_board, idx_map = SelfPlayGraveNN.canonicalise_board(
                        uniq_boards[u], int(uniq_players[u]), N
                    )
                    canon_stack[u] = canon_board
                    idx_maps[u] = idx_map

                encoded = torch.as_tensor(
                    canon_stack,
                    dtype=torch.long,
                    device=self.device,
                )

                policy_logits, value_t = self.net(encoded)
                probs_u = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()
                vals_u = value_t.view(-1).detach().cpu().numpy()

                for j, i in enumerate(non_idx):
                    u = int(inv[j])
                    leaf_board = non_boards[j]

                    raw_canon = probs_u[u]
                    raw_unmapped = raw_canon[idx_maps[u]]

                    empties = np.where(leaf_board == 0)[0]
                    if empties.size > 0:
                        pri = raw_unmapped[empties]
                        s = float(pri.sum())
                        if s > 0.0:
                            priors_batch[i, empties] = pri / s
                        else:
                            priors_batch[i, empties] = 1.0 / float(empties.size)
                    else:
                        priors_batch[i, :] = 1.0 / float(NN)

                    values_batch[i] = float(vals_u[u])

            self.tree.apply_evals_batch(priors_batch, values_batch)
            sims_done += B

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
            "batch =",
            self.batch_size,
        )

        return Move(action // N, action % N)