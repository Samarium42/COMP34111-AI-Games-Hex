import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import njit, prange
from numba.typed import Dict, List
from numba import types

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour

from agents.Group3.azalea_net import load_hex11_pretrained


class HexState:
    def __init__(self, board: Board, player: Colour):
        self.board_size = board.size
        self.board = Board(board_size=self.board_size)
        for x in range(self.board_size):
            for y in range(self.board_size):
                self.board.tiles[x][y].colour = board.tiles[x][y].colour

        self.player = player  

    def clone(self):
        return HexState(self.board, self.player)

    def legal_moves(self):
        N = self.board_size
        moves = []
        for x in range(N):
            for y in range(N):
                if self.board.tiles[x][y].colour is None:
                    moves.append(x * N + y)
        return moves

    def play(self, action_index: int):
        new_state = self.clone()
        x = action_index // self.board_size
        y = action_index % self.board_size
        new_state.board.set_tile_colour(x, y, self.player)
        new_state.player = Colour.RED if self.player == Colour.BLUE else Colour.BLUE
        return new_state

    def is_terminal(self):
        red_win = self.board.has_ended(Colour.RED)
        blue_win = self.board.has_ended(Colour.BLUE)
        return red_win or blue_win

    def result(self):
        """
        Return value from the perspective of the player to move in this state.

        +1 : good for the current player
        -1 : bad for the current player
         0 : draw / no winner (should not really occur in Hex)
        """
        winner = self.board.get_winner()
        if winner is None:
            return 0
        if winner == self.player:
            return 1
        else:
            return -1

    def encode(self, device=None, as_numpy=False):
        """
        Encode board for Azalea net:
            0 = empty
            1 = RED (X / first player)
            2 = BLUE (O / second player)

        Returns: tensor (N, N) of dtype long
        """
        N = self.board_size
        if device is None:
            device = torch.device("cpu")

        board_int = torch.zeros((N, N), dtype=torch.long, device=device)

        for x in range(N):
            for y in range(N):
                c = self.board.tiles[x][y].colour
                if c == Colour.RED:
                    board_int[x, y] = 1
                elif c == Colour.BLUE:
                    board_int[x, y] = 2
                else:
                    board_int[x, y] = 0

        if as_numpy:
            return board_int.cpu().numpy()
        return board_int


# ============================================================================
# NUMBA-ACCELERATED FUNCTIONS
# ============================================================================

@njit
def numba_select_child(children_N, children_W, children_prior, total_N, c_puct):
    """
    Numba-accelerated child selection using UCB formula.
    
    Args:
        children_N: array of visit counts for each child
        children_W: array of total values for each child
        children_prior: array of prior probabilities for each child
        total_N: total visit count of parent
        c_puct: exploration constant
    
    Returns:
        best_idx: index of best child
        best_score: UCB score of best child
    """
    n_children = len(children_N)
    best_idx = 0
    best_score = -1e9
    
    sqrt_total = np.sqrt(total_N + 1e-8)
    
    for i in range(n_children):
        Q = children_W[i] / (children_N[i] + 1e-8)
        U = c_puct * children_prior[i] * sqrt_total / (1 + children_N[i])
        score = Q + U
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx, best_score


@njit
def numba_backup_path(path_N, path_W, value):
    """
    Numba-accelerated backup along a path.
    
    Args:
        path_N: array of visit counts along path (modified in-place)
        path_W: array of total values along path (modified in-place)
        value: leaf value to backup
    """
    v = value
    for i in range(len(path_N)):
        path_N[i] += 1
        path_W[i] += v
        v = -v


@njit
def numba_mask_and_normalize(priors, legal_mask):
    """
    Numba-accelerated masking and normalization of policy priors.
    
    Args:
        priors: raw policy probabilities
        legal_mask: binary mask of legal moves
    
    Returns:
        normalized priors
    """
    masked = priors * legal_mask
    total = np.sum(masked)
    
    if total <= 0:
        # Uniform over legal moves
        n_legal = np.sum(legal_mask)
        if n_legal > 0:
            return legal_mask / n_legal
        else:
            return priors  # Shouldn't happen
    else:
        return masked / total


@njit
def numba_compute_legal_mask(board_flat, board_size):
    """
    Numba-accelerated legal move computation.
    
    Args:
        board_flat: flattened board array (N*N,)
        board_size: size of board
    
    Returns:
        mask: binary array where 1 = legal, 0 = illegal
    """
    mask = np.zeros(board_size * board_size, dtype=np.float32)
    for i in range(board_size * board_size):
        if board_flat[i] == 0:
            mask[i] = 1.0
    return mask


@njit(parallel=True)
def numba_aggregate_counts(all_action_indices, all_counts, board_size):
    """
    Numba-accelerated aggregation of visit counts from multiple MCTS runs.
    Uses parallel processing.
    
    Args:
        all_action_indices: list of action indices for each run
        all_counts: corresponding visit counts
        board_size: size of board
    
    Returns:
        aggregated counts array
    """
    total_counts = np.zeros(board_size * board_size, dtype=np.float32)
    
    for i in prange(len(all_action_indices)):
        action = all_action_indices[i]
        count = all_counts[i]
        total_counts[action] += count
    
    return total_counts


@njit
def numba_board_to_flat(board_2d):
    """
    Numba-accelerated board flattening.
    
    Args:
        board_2d: 2D board array (N, N)
    
    Returns:
        flattened array (N*N,)
    """
    return board_2d.flatten()


# ============================================================================
# NODE CLASS (with Numba-compatible data structures)
# ============================================================================

class Node:
    def __init__(self, state: HexState, parent, prior):
        self.state = state
        self.parent = parent
        self.prior = float(prior)

        self.children: dict[int, "Node"] = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

    def expand(self, legal, priors):
        for a in legal:
            p = float(priors[a])
            self.children[a] = Node(self.state.play(a), self, p)

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
        # Prepare data for Numba
        actions = list(self.children.keys())
        n = len(actions)
        
        children_N = np.zeros(n, dtype=np.float32)
        children_W = np.zeros(n, dtype=np.float32)
        children_prior = np.zeros(n, dtype=np.float32)
        
        for i, a in enumerate(actions):
            child = self.children[a]
            children_N[i] = child.N
            children_W[i] = child.W
            children_prior[i] = child.prior
        
        total_N = sum(child.N for child in self.children.values())
        
        # Call Numba-accelerated selection
        best_idx, best_score = numba_select_child(
            children_N, children_W, children_prior, total_N, c_puct
        )
        
        best_a = actions[best_idx]
        best_child = self.children[best_a]
        
        return best_a, best_child


# ============================================================================
# MCTS WITH NUMBA ACCELERATION
# ============================================================================

class MCTS:
    def __init__(self, net, sims=500, c_puct=1.2, device="mps"):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.device = device

    def run(self, root_state: HexState):
        root = Node(root_state, None, 1.0)
        self.expand_and_eval(root)

        for _ in range(self.sims):
            node = root

            while node.children and not node.state.is_terminal():
                _, node = node.select_child(self.c_puct)

            if node.state.is_terminal():
                value = float(node.state.result())
                node.backup(value)
                continue

            value = self.expand_and_eval(node)
            node.backup(value)

        # Use Numba to aggregate counts
        N = root.state.board_size
        counts = np.zeros(N * N, dtype=np.float32)
        for a, child in root.children.items():
            counts[a] = child.N

        return counts

    @torch.no_grad()
    def expand_and_eval(self, node: Node):
        if node.state.is_terminal():
            return float(node.state.result())

        x = node.state.encode(device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        logits = logits[0]
        value = value.item()

        priors = torch.softmax(logits, dim=0).detach().cpu().numpy()

        legal = node.state.legal_moves()
        
        # Use Numba for masking and normalization
        N = node.state.board_size
        legal_mask = np.zeros(N * N, dtype=np.float32)
        for a in legal:
            legal_mask[a] = 1.0
        
        priors = numba_mask_and_normalize(priors, legal_mask)

        node.expand(legal, priors)
        return value


# ============================================================================
# RESNET BLOCKS (unchanged)
# ============================================================================

class ResNetBlock(nn.Module):
    def __init__(self, channels, reach=1, scale=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=2 * reach + 1, padding=reach, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.scale = scale

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.scale * out + residual
        return out * torch.sigmoid(out)


class HexResNet(nn.Module):
    def __init__(self, board_size=11, in_channels=4, channels=32, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(channels)

        self.trunk = nn.Sequential(
            *[ResNetBlock(channels, reach=1) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(
            2 * board_size * board_size, board_size * board_size
        )

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        B = x.shape[0]
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.trunk(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(B, -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(B, -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return policy_logits, v


# ============================================================================
# NUMBA GRAVENN AGENT
# ============================================================================

class NumbaGraveNN(AgentBase):
    def __init__(self, colour: Colour,
                 load_path="models/hex11-20180712-3362.policy.pth",
                 use_azalea=True):
        super().__init__(colour)
        self.board_size = 11
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_azalea:
            print(f"[NumbaGraveNN] Using Azalea pretrained Hex network from {load_path}")
            self.net = load_hex11_pretrained(load_path, self.device, board_size=self.board_size)
        else:
            # fallback network
            self.net = HexResNet(board_size=self.board_size, in_channels=4).to(self.device)
            try:
                state_dict = torch.load(load_path, map_location=self.device)
                self.net.load_state_dict(state_dict)
                print(f"[NumbaGraveNN] Loaded GraveNN weights from {load_path}")
            except FileNotFoundError:
                print(f"[NumbaGraveNN] No checkpoint at {load_path}, using random weights")

        self.net.eval()
        self.mcts = MCTS(self.net, sims=500, device=self.device)
        
        print("[NumbaGraveNN] Numba acceleration enabled for MCTS operations")

    def make_move(self, colour: Colour, board: Board, opponent_move: Move | None) -> Move:
        """
        Called by Game.py as:
            make_move(self.turn, playerBoard, opponentMove)

        colour: Colour.RED or Colour.BLUE for the player to move
        board:  the current Board instance
        opponent_move: the last Move made by the opponent (can be None on first turn)
        """
        N = board.size

        # Build our HexState from the current board and player to move
        state = HexState(board, colour)

        import time 
        start_time = time.time()   

        # Run MCTS guided by the neural net (with Numba acceleration)
        counts = self.mcts.run(state)

        end_time = time.time()
        print(f"[NumbaGraveNN] MCTS completed in {end_time - start_time:.2f} seconds.")

        # Pick the move with the highest visit count
        action = int(counts.argmax())

        x = action // N
        y = action % N

        return Move(x, y)