"""
NumbaGraveNN - Optimized MCTS + Neural Network Agent (No Numba)

Optimizations retained:
- Flat numpy board representation
- Fast cloning with object.__new__()
- Tree reuse between moves
- Vectorized operations
"""

import torch
import numpy as np

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour

from agents.Group3.azalea_net import load_hex11_pretrained


class HexState:
    """Fast Hex state using flat numpy array."""
    __slots__ = ['N', 'player', 'board_flat']
    
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
        """Fast clone bypassing __init__."""
        st = object.__new__(HexState)
        st.N = self.N
        st.player = self.player
        st.board_flat = self.board_flat.copy()
        return st

    def legal_moves(self):
        """Vectorized legal move computation."""
        return np.where(self.board_flat == 0)[0].tolist()

    def play(self, action_idx: int):
        """Return new state with move applied."""
        st = self.clone()
        st.board_flat[action_idx] = 1 if self.player == Colour.RED else 2
        st.player = Colour.RED if self.player == Colour.BLUE else Colour.BLUE
        return st

    def to_Board(self):
        """Convert to Board object (only for terminal checking)."""
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
        """Return +1 if current player wins, -1 if loses."""
        b = self.to_Board()
        winner = b.get_winner()
        if winner is None:
            return 0
        return +1 if winner == self.player else -1

    def encode(self, device=None):
        """Returns (N, N) tensor for Azalea network."""
        return torch.tensor(
            self.board_flat.reshape(self.N, self.N),
            dtype=torch.long,
            device=device
        )

    def equals(self, other):
        """Check state equality for tree reuse."""
        return (
            self.player == other.player and
            np.array_equal(self.board_flat, other.board_flat)
        )


def select_child_idx(children_N, children_W, children_prior, total_N, c_puct):
    """Select best child using PUCT formula."""
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
    return best_idx


def mask_and_normalize(priors, mask):
    """Mask illegal moves and normalize probabilities."""
    out = priors * mask
    s = np.sum(out)
    if s <= 0:
        legal = np.sum(mask)
        if legal > 0:
            return mask / legal
        return priors
    return out / s


class Node:
    """MCTS tree node."""
    __slots__ = ['state', 'parent', 'prior', 'children', 'N', 'W', 'Q']
    
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
        idx = select_child_idx(N_arr, W_arr, P_arr, total, c_puct)
        a = actions[idx]
        return a, self.children[a]


class MCTS:
    """Monte Carlo Tree Search with tree reuse."""
    
    def __init__(self, net, sims=300, c_puct=1.2, device="cpu"):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.device = device
        self.root = None

    def run(self, root_state: HexState):
        # Tree reuse: check if we can reuse previous tree
        if self.root is None:
            self.root = Node(root_state, None, 1.0)
        elif not self.root.state.equals(root_state):
            self.root = Node(root_state, None, 1.0)
        else:
            self.root.parent = None

        # Initial expansion if needed
        if not self.root.children:
            v = self.expand_and_eval(self.root)
            self.root.backup(v)

        # MCTS iterations
        for _ in range(self.sims):
            node = self.root

            # Selection
            while node.children and not node.state.is_terminal():
                _, node = node.select_child(self.c_puct)

            # Terminal node
            if node.state.is_terminal():
                val = float(node.state.result())
                node.backup(val)
                continue

            # Expansion and evaluation
            val = self.expand_and_eval(node)
            node.backup(val)

        # Collect visit counts
        N = root_state.N
        counts = np.zeros(N * N, dtype=np.float32)
        for a, child in self.root.children.items():
            counts[a] = child.N
        return counts

    @torch.no_grad()
    def expand_and_eval(self, node: Node):
        """Expand node and return value estimate from neural network."""
        if node.state.is_terminal():
            return float(node.state.result())

        x = node.state.encode(self.device).unsqueeze(0)
        logits, value = self.net(x)
        value = value.item()

        pri_raw = torch.softmax(logits[0], dim=0).cpu().numpy()
        legal = node.state.legal_moves()

        # Create mask for legal moves
        mask = np.zeros(node.state.N * node.state.N, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0

        pri = mask_and_normalize(pri_raw, mask)
        node.expand(legal, pri)

        return value

    def advance_root(self, action):
        """Advance tree root after a move (for tree reuse)."""
        if self.root and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None


class NumbaGraveNN(AgentBase):
    """
    Optimized Hex agent using MCTS + Neural Network.
    
    Despite the name, this version does NOT require Numba.
    """
    
    def __init__(self, colour: Colour,
                 load_path="models/hex11-20180712-3362.policy.pth",
                 sims=300):
        super().__init__(colour)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[NumbaGraveNN] Loading Azalea model, device={self.device}")
        self.net = load_hex11_pretrained(load_path, self.device)
        self.net.eval()
        
        self.mcts = MCTS(self.net, sims=sims, device=self.device)
        print(f"[NumbaGraveNN] Ready: sims={sims}")

    def make_move(self, turn: int, board: Board, opponent_move: Move | None) -> Move:
        N = board.size
        state = HexState(board, self.colour)

        import time
        t0 = time.time()
        counts = self.mcts.run(state)
        print(f"[NumbaGraveNN] MCTS took {time.time() - t0:.2f}s")

        action = int(np.argmax(counts))

        # Tree reuse: advance root to chosen action
        self.mcts.advance_root(action)

        return Move(action // N, action % N)