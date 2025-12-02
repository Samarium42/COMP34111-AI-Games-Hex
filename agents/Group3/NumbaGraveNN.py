import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import njit, prange

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour

from agents.Group3.azalea_net import load_hex11_pretrained

class HexState:
    def __init__(self, board: Board, player: Colour):
        self.N = board.size
        self.player = player

        arr = np.zeros(self.N * self.N, dtype=np.int8)
        for x in range(self.N):
            for y in range(self.N):
                c = board.tiles[x][y].colour
                if c == Colour.RED:
                    arr[x * self.N + y] = 1
                elif c == Colour.BLUE:
                    arr[x * self.N + y] = 2
        self.board_flat = arr

    def clone(self):
        st = object.__new__(HexState)
        st.N = self.N
        st.player = self.player
        st.board_flat = self.board_flat.copy()
        return st

    def legal_moves(self):
        return np.where(self.board_flat == 0)[0].tolist()

    def play(self, action_idx: int):
        st = self.clone()
        st.board_flat[action_idx] = 1 if self.player == Colour.RED else 2
        st.player = Colour.RED if self.player == Colour.BLUE else Colour.BLUE
        return st

    def to_Board(self):
        """Only used for terminal-state checking."""
        b = Board(board_size=self.N)
        for i in range(self.N * self.N):
            v = self.board_flat[i]
            if v == 1:
                b.tiles[i // self.N][i % self.N].colour = Colour.RED
            elif v == 2:
                b.tiles[i // self.N][i % self.N].colour = Colour.BLUE
        return b

    def is_terminal(self):
        b = self.to_Board()
        return b.has_ended(Colour.RED) or b.has_ended(Colour.BLUE)

    def result(self):
        b = self.to_Board()
        winner = b.get_winner()
        if winner is None:
            return 0
        return +1 if winner == self.player else -1

    def encode(self, device=None):
        """
        Returns (N, N) tensor for Azalea:
        0 empty, 1 red, 2 blue
        """
        return torch.tensor(
            self.board_flat.reshape(self.N, self.N),
            dtype=torch.long,
            device=device
        )

    def equals(self, other):
        return (
            self.player == other.player and
            np.array_equal(self.board_flat, other.board_flat)
        )



@njit
def numba_select_child(children_N, children_W, children_prior, total_N, c_puct):
    best_idx = 0
    best_score = -1e12
    sqrt_total = np.sqrt(total_N + 1e-8)

    for i in range(len(children_N)):
        Q = children_W[i] / (children_N[i] + 1e-8)
        U = c_puct * children_prior[i] * sqrt_total / (1 + children_N[i])
        score = Q + U
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


@njit
def numba_mask_and_normalize(priors, mask):
    out = priors * mask
    s = np.sum(out)
    if s <= 0:
        legal = np.sum(mask)
        if legal > 0:
            return mask / legal
        return priors
    return out / s


@njit(parallel=True)
def numba_compute_legal_mask(board_flat, N):
    mask = np.zeros(N * N, dtype=np.float32)
    for i in prange(N * N):
        if board_flat[i] == 0:
            mask[i] = 1.0
    return mask


class Node:
    def __init__(self, state: HexState, parent, prior):
        self.state = state
        self.parent = parent
        self.prior = float(prior)

        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

    def expand(self, legal, priors):
        for a in legal:
            self.children[a] = Node(self.state.play(a), self, float(priors[a]))

    def backup(self, value):
        node = self
        v = value
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v
            node = node.parent

    def select_child(self, c_puct):
        actions = list(self.children.keys())
        n = len(actions)

        N_arr = np.zeros(n, dtype=np.float32)
        W_arr = np.zeros(n, dtype=np.float32)
        P_arr = np.zeros(n, dtype=np.float32)

        for i, a in enumerate(actions):
            ch = self.children[a]
            N_arr[i] = ch.N
            W_arr[i] = ch.W
            P_arr[i] = ch.prior

        total = np.sum(N_arr)

        idx, _ = numba_select_child(N_arr, W_arr, P_arr, total, c_puct)
        a = actions[idx]
        return a, self.children[a]


class MCTS:
    def __init__(self, net, sims=300, c_puct=1.2, device="cpu"):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.device = device
        self.root = None

    def run(self, root_state: HexState):

        if self.root is None:
            self.root = Node(root_state, None, 1.0)
        elif not self.root.state.equals(root_state):
            self.root = Node(root_state, None, 1.0)
        else:
            self.root.parent = None

        if not self.root.children:
            v = self.expand_and_eval(self.root)
            self.root.backup(v)

        for _ in range(self.sims):
            node = self.root

            while node.children and not node.state.is_terminal():
                _, node = node.select_child(self.c_puct)

            if node.state.is_terminal():
                val = float(node.state.result())
                node.backup(val)
                continue

            val = self.expand_and_eval(node)
            node.backup(val)

        N = root_state.N
        counts = np.zeros(N * N, dtype=np.float32)
        for a, child in self.root.children.items():
            counts[a] = child.N
        return counts

    @torch.no_grad()
    def expand_and_eval(self, node: Node):
        if node.state.is_terminal():
            return float(node.state.result())

        x = node.state.encode(self.device).unsqueeze(0)
        logits, value = self.net(x)
        value = value.item()

        pri_raw = torch.softmax(logits[0], dim=0).cpu().numpy()
        legal = node.state.legal_moves()

        mask = np.zeros(node.state.N * node.state.N, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0

        pri = numba_mask_and_normalize(pri_raw, mask)
        node.expand(legal, pri)

        return value

class NumbaGraveNN(AgentBase):
    def __init__(self, colour: Colour,
                 load_path="models/hex11-20180712-3362.policy.pth",
                 use_azalea=True):
        super().__init__(colour)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_azalea:
            print("[NumbaGraveNN] Loading Azalea pretrained model")
            self.net = load_hex11_pretrained(load_path, self.device)
        else:
            self.net = HexResNet().to(self.device)

        self.net.eval()
        self.mcts = MCTS(self.net, sims=300, device=self.device)

    def make_move(self, colour: Colour, board: Board, opponent_move: Move | None):
        state = HexState(board, colour)

        import time
        t0 = time.time()
        counts = self.mcts.run(state)
        print(f"[NumbaGraveNN] MCTS took {time.time() - t0:.2f}s")

        action = int(np.argmax(counts))

        # tree reuse
        if action in self.mcts.root.children:
            self.mcts.root = self.mcts.root.children[action]
            self.mcts.root.parent = None
        else:
            self.mcts.root = None

        N = board.size
        return Move(action // N, action % N)
