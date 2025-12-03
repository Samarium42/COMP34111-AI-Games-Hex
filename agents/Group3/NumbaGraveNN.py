"""
OptimizedHexAgent - Fast MCTS + Neural Network with Tactical Heuristics

Optimizations:
- Flat numpy board representation (fast cloning)
- Tree reuse between moves
- Vectorized operations
- Numba JIT compilation

Heuristics (applied to top-K candidates only):
1. Bridge Detection (Very High impact, Low cost)
2. Virtual Connections (High impact, Medium cost)
3. Ladder Detection (High impact, Medium cost)
"""

import torch
import numpy as np
from numba import njit, prange

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour

from agents.Group3.azalea_net import load_hex11_pretrained


# ============================================================================
# CONSTANTS FOR HEX GRID
# ============================================================================

# Neighbour offsets for hex grid (6 directions)
NEIGHBOUR_DX = np.array([-1, -1, 0, 0, 1, 1], dtype=np.int32)
NEIGHBOUR_DY = np.array([0, 1, -1, 1, -1, 0], dtype=np.int32)

# Bridge patterns: (dx, dy, guard1_dx, guard1_dy, guard2_dx, guard2_dy)
# A bridge connects stone at (x,y) to stone at (x+dx, y+dy)
# The bridge is secure if both guard cells are empty
BRIDGE_PATTERNS = np.array([
    [-1, 1, -1, 0, 0, 1],   # upper-right bridge
    [1, -1, 1, 0, 0, -1],   # lower-left bridge
    [-1, 0, -1, 1, 0, -1],  # upper-left bridge
    [1, 0, 1, -1, 0, 1],    # lower-right bridge
    [0, 1, -1, 1, 1, 0],    # right bridge
    [0, -1, 1, -1, -1, 0],  # left bridge
], dtype=np.int32)


# ============================================================================
# FAST HEX STATE (Flat numpy array representation)
# ============================================================================

class HexState:
    """
    Fast Hex game state using flat numpy array.
    board_flat: 0=empty, 1=RED, 2=BLUE
    """
    __slots__ = ['N', 'player', 'board_flat']
    
    def __init__(self, board: Board, player: Colour):
        self.N = board.size
        self.player = player
        
        # Convert to flat numpy array
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
        """Return +1 if current player wins, -1 if loses, 0 for draw."""
        b = self.to_Board()
        winner = b.get_winner()
        if winner is None:
            return 0
        return +1 if winner == self.player else -1

    def encode(self, device=None):
        """Encode for neural network: (N, N) tensor."""
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

    def get_2d_board(self) -> np.ndarray:
        """Get 2D board view for heuristics."""
        return self.board_flat.reshape(self.N, self.N)


# ============================================================================
# BRIDGE DETECTION HEURISTIC
# ============================================================================

@njit
def count_bridges(board_flat: np.ndarray, N: int, player: int) -> int:
    """
    Count bridge connections for a player.
    Works with flat board array.
    """
    count = 0
    
    for x in range(N):
        for y in range(N):
            idx = x * N + y
            if board_flat[idx] != player:
                continue
            
            for p in range(6):
                dx = BRIDGE_PATTERNS[p, 0]
                dy = BRIDGE_PATTERNS[p, 1]
                g1x = BRIDGE_PATTERNS[p, 2]
                g1y = BRIDGE_PATTERNS[p, 3]
                g2x = BRIDGE_PATTERNS[p, 4]
                g2y = BRIDGE_PATTERNS[p, 5]
                
                # Target cell
                nx, ny = x + dx, y + dy
                # Guard cells
                gx1, gy1 = x + g1x, y + g1y
                gx2, gy2 = x + g2x, y + g2y
                
                # Bounds checking
                if not (0 <= nx < N and 0 <= ny < N):
                    continue
                if not (0 <= gx1 < N and 0 <= gy1 < N):
                    continue
                if not (0 <= gx2 < N and 0 <= gy2 < N):
                    continue
                
                # Target has our stone, both guards empty
                target_idx = nx * N + ny
                guard1_idx = gx1 * N + gy1
                guard2_idx = gx2 * N + gy2
                
                if (board_flat[target_idx] == player and 
                    board_flat[guard1_idx] == 0 and 
                    board_flat[guard2_idx] == 0):
                    count += 1
    
    # Each bridge counted twice (once from each end)
    return count // 2


@njit
def bridge_heuristic(board_flat: np.ndarray, N: int, player: int, action: int) -> float:
    """
    Score a move based on bridge creation/destruction.
    """
    opp = 3 - player
    
    # Count bridges before
    our_before = count_bridges(board_flat, N, player)
    opp_before = count_bridges(board_flat, N, opp)
    
    # Simulate move
    board_flat[action] = player
    
    # Count bridges after
    our_after = count_bridges(board_flat, N, player)
    opp_after = count_bridges(board_flat, N, opp)
    
    # Undo move
    board_flat[action] = 0
    
    # Score: creating bridges + breaking opponent bridges
    our_delta = our_after - our_before
    opp_delta = opp_before - opp_after
    
    return our_delta * 2.0 + opp_delta * 1.5


# ============================================================================
# VIRTUAL CONNECTIONS HEURISTIC
# ============================================================================

@njit
def find_root(parent: np.ndarray, x: int) -> int:
    """Union-Find: find root with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression
    while parent[x] != root:
        next_x = parent[x]
        parent[x] = root
        x = next_x
    return root


@njit
def union(parent: np.ndarray, rank: np.ndarray, a: int, b: int):
    """Union-Find: union by rank."""
    ra = find_root(parent, a)
    rb = find_root(parent, b)
    if ra != rb:
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1


@njit
def compute_virtual_components(board_flat: np.ndarray, N: int, player: int) -> np.ndarray:
    """
    Compute virtually connected components using Union-Find.
    Considers both direct adjacency and bridge connections.
    """
    size = N * N
    parent = np.arange(size, dtype=np.int32)
    rank = np.zeros(size, dtype=np.int32)
    
    # Pass 1: Direct adjacencies
    for x in range(N):
        for y in range(N):
            idx = x * N + y
            if board_flat[idx] != player:
                continue
            
            for i in range(6):
                nx = x + NEIGHBOUR_DX[i]
                ny = y + NEIGHBOUR_DY[i]
                if 0 <= nx < N and 0 <= ny < N:
                    nidx = nx * N + ny
                    if board_flat[nidx] == player:
                        union(parent, rank, idx, nidx)
    
    # Pass 2: Bridge connections
    for x in range(N):
        for y in range(N):
            idx = x * N + y
            if board_flat[idx] != player:
                continue
            
            for p in range(6):
                dx = BRIDGE_PATTERNS[p, 0]
                dy = BRIDGE_PATTERNS[p, 1]
                g1x = BRIDGE_PATTERNS[p, 2]
                g1y = BRIDGE_PATTERNS[p, 3]
                g2x = BRIDGE_PATTERNS[p, 4]
                g2y = BRIDGE_PATTERNS[p, 5]
                
                nx, ny = x + dx, y + dy
                gx1, gy1 = x + g1x, y + g1y
                gx2, gy2 = x + g2x, y + g2y
                
                if not (0 <= nx < N and 0 <= ny < N):
                    continue
                if not (0 <= gx1 < N and 0 <= gy1 < N):
                    continue
                if not (0 <= gx2 < N and 0 <= gy2 < N):
                    continue
                
                nidx = nx * N + ny
                g1idx = gx1 * N + gy1
                g2idx = gx2 * N + gy2
                
                if (board_flat[nidx] == player and 
                    board_flat[g1idx] == 0 and 
                    board_flat[g2idx] == 0):
                    union(parent, rank, idx, nidx)
    
    # Finalize roots
    for i in range(size):
        find_root(parent, i)
    
    return parent


@njit
def count_components(parent: np.ndarray, board_flat: np.ndarray, N: int, player: int) -> int:
    """Count unique connected components."""
    size = N * N
    seen = np.zeros(size, dtype=np.int32)
    num_seen = 0
    
    for i in range(size):
        if board_flat[i] == player:
            root = find_root(parent, i)
            found = False
            for j in range(num_seen):
                if seen[j] == root:
                    found = True
                    break
            if not found:
                seen[num_seen] = root
                num_seen += 1
    
    return num_seen


@njit
def check_edge_connection(parent: np.ndarray, board_flat: np.ndarray, N: int, player: int) -> tuple:
    """
    Check edge connections for a player.
    Returns (connected_to_start, connected_to_end, has_winning_path)
    """
    start_roots = np.zeros(N, dtype=np.int32)
    end_roots = np.zeros(N, dtype=np.int32)
    num_start = 0
    num_end = 0
    
    if player == 1:  # RED: top (x=0) to bottom (x=N-1)
        for y in range(N):
            # Top edge
            idx_top = 0 * N + y
            if board_flat[idx_top] == player:
                root = find_root(parent, idx_top)
                found = False
                for i in range(num_start):
                    if start_roots[i] == root:
                        found = True
                        break
                if not found:
                    start_roots[num_start] = root
                    num_start += 1
            
            # Bottom edge
            idx_bot = (N - 1) * N + y
            if board_flat[idx_bot] == player:
                root = find_root(parent, idx_bot)
                found = False
                for i in range(num_end):
                    if end_roots[i] == root:
                        found = True
                        break
                if not found:
                    end_roots[num_end] = root
                    num_end += 1
    else:  # BLUE: left (y=0) to right (y=N-1)
        for x in range(N):
            # Left edge
            idx_left = x * N + 0
            if board_flat[idx_left] == player:
                root = find_root(parent, idx_left)
                found = False
                for i in range(num_start):
                    if start_roots[i] == root:
                        found = True
                        break
                if not found:
                    start_roots[num_start] = root
                    num_start += 1
            
            # Right edge
            idx_right = x * N + (N - 1)
            if board_flat[idx_right] == player:
                root = find_root(parent, idx_right)
                found = False
                for i in range(num_end):
                    if end_roots[i] == root:
                        found = True
                        break
                if not found:
                    end_roots[num_end] = root
                    num_end += 1
    
    connected_start = num_start > 0
    connected_end = num_end > 0
    
    # Check for winning path (same root in both)
    has_win = False
    for i in range(num_start):
        for j in range(num_end):
            if start_roots[i] == end_roots[j]:
                has_win = True
                break
        if has_win:
            break
    
    return connected_start, connected_end, has_win


@njit
def virtual_connection_heuristic(board_flat: np.ndarray, N: int, player: int, action: int) -> float:
    """
    Score based on virtual connectivity improvements.
    """
    # Before move
    parent_before = compute_virtual_components(board_flat, N, player)
    comp_before = count_components(parent_before, board_flat, N, player)
    start_before, end_before, win_before = check_edge_connection(parent_before, board_flat, N, player)
    
    # Simulate move
    board_flat[action] = player
    
    # After move
    parent_after = compute_virtual_components(board_flat, N, player)
    comp_after = count_components(parent_after, board_flat, N, player)
    start_after, end_after, win_after = check_edge_connection(parent_after, board_flat, N, player)
    
    # Undo move
    board_flat[action] = 0
    
    score = 0.0
    
    # Reward reducing components (connecting groups)
    comp_reduction = comp_before - comp_after
    score += comp_reduction * 3.0
    
    # Reward connecting to edges
    if not start_before and start_after:
        score += 2.0
    if not end_before and end_after:
        score += 2.0
    
    # Big reward for virtual win
    if not win_before and win_after:
        score += 10.0
    
    return score


# ============================================================================
# LADDER DETECTION HEURISTIC
# ============================================================================

@njit
def detect_ladder_threat(board_flat: np.ndarray, N: int, player: int, max_depth: int = 8) -> tuple:
    """
    Detect if opponent has a ladder threat reaching their goal.
    Returns (has_threat, blocking_moves, num_blocks)
    """
    opp = 3 - player
    blocking_moves = np.zeros(max_depth, dtype=np.int32)
    
    # Ladder directions based on opponent
    if opp == 1:  # RED goes top to bottom
        ladder_dirs_x = np.array([1, 1], dtype=np.int32)
        ladder_dirs_y = np.array([-1, 0], dtype=np.int32)
    else:  # BLUE goes left to right
        ladder_dirs_x = np.array([0, -1], dtype=np.int32)
        ladder_dirs_y = np.array([1, 1], dtype=np.int32)
    
    # Search for ladder starts
    for start_x in range(N):
        for start_y in range(N):
            start_idx = start_x * N + start_y
            if board_flat[start_idx] != opp:
                continue
            
            # Try each ladder direction
            for d in range(2):
                ldx = ladder_dirs_x[d]
                ldy = ladder_dirs_y[d]
                
                x, y = start_x, start_y
                temp_blocks = np.zeros(max_depth, dtype=np.int32)
                temp_count = 0
                
                for step in range(max_depth):
                    nx, ny = x + ldx, y + ldy
                    
                    if not (0 <= nx < N and 0 <= ny < N):
                        break
                    
                    nidx = nx * N + ny
                    
                    if board_flat[nidx] == opp:
                        # Opponent stone, ladder continues
                        x, y = nx, ny
                    elif board_flat[nidx] == 0:
                        # Empty cell, potential ladder extension
                        temp_blocks[temp_count] = nidx
                        temp_count += 1
                        
                        # Check if reaches goal
                        if opp == 1 and nx == N - 1:  # RED reached bottom
                            for i in range(temp_count):
                                blocking_moves[i] = temp_blocks[i]
                            return True, blocking_moves, temp_count
                        if opp == 2 and ny == N - 1:  # BLUE reached right
                            for i in range(temp_count):
                                blocking_moves[i] = temp_blocks[i]
                            return True, blocking_moves, temp_count
                        
                        x, y = nx, ny
                    else:
                        # Our stone blocks ladder
                        break
    
    return False, blocking_moves, 0


@njit
def ladder_heuristic(board_flat: np.ndarray, N: int, player: int, action: int) -> float:
    """
    Score based on ladder blocking/creation.
    """
    opp = 3 - player
    score = 0.0
    
    # Check if move blocks opponent ladder
    has_threat, blocking_moves, num_blocks = detect_ladder_threat(board_flat, N, player, 8)
    
    if has_threat and num_blocks > 0:
        for i in range(num_blocks):
            if action == blocking_moves[i]:
                # Earlier blocks more valuable
                block_score = 5.0 - i * 0.5
                if block_score < 1.0:
                    block_score = 1.0
                score += block_score
                break
    
    # Check if we create a ladder threat
    board_flat[action] = player
    opp_threat, _, _ = detect_ladder_threat(board_flat, N, opp, 8)
    board_flat[action] = 0
    
    if opp_threat:
        score += 3.0
    
    return score


# ============================================================================
# COMBINED HEURISTIC EVALUATION
# ============================================================================

@njit
def evaluate_move_heuristics(board_flat: np.ndarray, N: int, player: int, action: int) -> float:
    """
    Compute all heuristics for a single move.
    Returns weighted sum of heuristic scores.
    """
    # Weights (tunable)
    W_BRIDGE = 1.0
    W_VIRTUAL = 0.8
    W_LADDER = 1.2
    
    # Make a copy since heuristics modify board temporarily
    board_copy = board_flat.copy()
    
    h_bridge = bridge_heuristic(board_copy, N, player, action)
    h_virtual = virtual_connection_heuristic(board_copy, N, player, action)
    h_ladder = ladder_heuristic(board_copy, N, player, action)
    
    return W_BRIDGE * h_bridge + W_VIRTUAL * h_virtual + W_LADDER * h_ladder


@njit(parallel=True)
def evaluate_top_k_moves(board_flat: np.ndarray, N: int, player: int, 
                         top_actions: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """
    Evaluate heuristics for top-K moves in parallel.
    Returns array of final scores.
    """
    k = len(top_actions)
    scores = np.zeros(k, dtype=np.float32)
    
    for i in prange(k):
        action = top_actions[i]
        base_score = counts[action]
        
        # Each thread gets its own board copy
        board_copy = board_flat.copy()
        h_score = evaluate_move_heuristics(board_copy, N, player, action)
        
        scores[i] = base_score + h_score
    
    return scores


# ============================================================================
# MCTS COMPONENTS
# ============================================================================

@njit
def numba_select_child(children_N, children_W, children_prior, total_N, c_puct):
    """Numba-accelerated PUCT selection."""
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
    """Mask and normalize policy priors."""
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
    
    def __init__(self, state: HexState, parent, prior: float):
        self.state = state
        self.parent = parent
        self.prior = prior
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
        
        return actions[idx], self.children[actions[idx]]


class MCTS:
    """Monte Carlo Tree Search with tree reuse."""
    
    def __init__(self, net, sims: int = 300, c_puct: float = 1.2, device: str = "cpu"):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.device = device
        self.root = None

    def run(self, root_state: HexState) -> np.ndarray:
        """Run MCTS and return visit counts."""
        
        # Tree reuse: check if we can reuse previous tree
        if self.root is None:
            self.root = Node(root_state, None, 1.0)
        elif not self.root.state.equals(root_state):
            self.root = Node(root_state, None, 1.0)
        else:
            self.root.parent = None  # Reuse tree
        
        # Initial expansion if needed
        if not self.root.children:
            v = self._expand_and_eval(self.root)
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
            val = self._expand_and_eval(node)
            node.backup(val)
        
        # Collect visit counts
        N = root_state.N
        counts = np.zeros(N * N, dtype=np.float32)
        for a, child in self.root.children.items():
            counts[a] = child.N
        
        return counts

    @torch.no_grad()
    def _expand_and_eval(self, node: Node) -> float:
        """Expand node and return value estimate."""
        if node.state.is_terminal():
            return float(node.state.result())
        
        # Neural network forward pass
        x = node.state.encode(self.device).unsqueeze(0)
        logits, value = self.net(x)
        value = value.item()
        
        # Get policy priors
        pri_raw = torch.softmax(logits[0], dim=0).cpu().numpy()
        legal = node.state.legal_moves()
        
        # Mask and normalize
        mask = np.zeros(node.state.N * node.state.N, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        
        pri = numba_mask_and_normalize(pri_raw, mask)
        node.expand(legal, pri)
        
        return value

    def advance_root(self, action: int):
        """Advance tree root after a move (for tree reuse)."""
        if self.root and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = None


# ============================================================================
# MAIN AGENT CLASS
# ============================================================================

class NumbaGraveNN(AgentBase):
    """
    Optimized Hex agent with MCTS + NN + Tactical Heuristics.
    
    Features:
    - Fast flat board representation
    - Tree reuse between moves
    - Numba-accelerated heuristics
    - Top-K filtering for heuristic evaluation
    """
    
    def __init__(self, colour: Colour,
                 model_path: str = "models/hex11-20180712-3362.policy.pth",
                 sims: int = 300,
                 c_puct: float = 1.2,
                 use_heuristics: bool = True,
                 top_k: int = 8):
        super().__init__(colour)
        self.use_heuristics = use_heuristics
        self.top_k = top_k
        
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[OptimizedHexAgent] Using CUDA GPU")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("[OptimizedHexAgent] Using Apple MPS")
        else:
            self.device = torch.device("cpu")
            print("[OptimizedHexAgent] Using CPU")
        
        # Load neural network
        print(f"[OptimizedHexAgent] Loading model from {model_path}")
        self.net = load_hex11_pretrained(model_path, self.device, board_size=11)
        self.net.eval()
        
        # Initialize MCTS
        self.mcts = MCTS(self.net, sims=sims, c_puct=c_puct, device=self.device)
        
        print(f"[OptimizedHexAgent] Initialized: sims={sims}, heuristics={use_heuristics}, top_k={top_k}")

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a move given current board state."""
        N = board.size
        player = 1 if self.colour == Colour.RED else 2
        
        # Build state
        state = HexState(board, self.colour)
        
        # Run MCTS
        import time
        t0 = time.time()
        counts = self.mcts.run(state)
        
        # Apply heuristics to top-K candidates
        if self.use_heuristics:
            best_action = self._select_with_heuristics(state, counts, player)
        else:
            best_action = int(np.argmax(counts))
        
        t1 = time.time()
        print(f"[OptimizedHexAgent] Move took {t1-t0:.2f}s")
        
        # Advance tree for reuse
        self.mcts.advance_root(best_action)
        
        return Move(best_action // N, best_action % N)

    def _select_with_heuristics(self, state: HexState, counts: np.ndarray, player: int) -> int:
        """Select best move using MCTS counts + heuristics."""
        N = state.N
        
        # Get top-K candidates by visit count
        k = min(self.top_k, int(np.sum(counts > 0)))
        if k == 0:
            return int(np.argmax(counts))
        
        top_indices = np.argsort(counts)[-k:]
        top_actions = np.array([idx for idx in top_indices if counts[idx] > 0], dtype=np.int32)
        
        if len(top_actions) == 0:
            return int(np.argmax(counts))
        
        # Evaluate heuristics for top candidates
        scores = evaluate_top_k_moves(
            state.board_flat.copy(),
            N,
            player,
            top_actions,
            counts
        )
        
        # Return action with best combined score
        best_idx = np.argmax(scores)
        return int(top_actions[best_idx])


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing OptimizedHexAgent...")
    print()
    
    # Test heuristics compilation
    print("Compiling Numba functions...")
    board_flat = np.zeros(121, dtype=np.int8)
    board_flat[60] = 1  # Center stone
    board_flat[50] = 1  # Another stone
    
    # Test bridge
    bridges = count_bridges(board_flat, 11, 1)
    print(f"Bridge count: {bridges}")
    
    # Test bridge heuristic
    h_bridge = bridge_heuristic(board_flat.copy(), 11, 1, 70)
    print(f"Bridge heuristic for (6,4): {h_bridge}")
    
    # Test virtual connection
    h_virtual = virtual_connection_heuristic(board_flat.copy(), 11, 1, 70)
    print(f"Virtual connection heuristic: {h_virtual}")
    
    # Test ladder
    board_flat[25] = 2  # Opponent stone
    has_ladder, blocks, n = detect_ladder_threat(board_flat, 11, 1, 8)
    print(f"Ladder threat: {has_ladder}, blocks: {n}")
    
    h_ladder = ladder_heuristic(board_flat.copy(), 11, 1, 70)
    print(f"Ladder heuristic: {h_ladder}")
    
    print()
    print("All Numba functions compiled successfully!")
    print()
    
    # Test full agent (requires model file)
    try:
        board = Board(11)
        agent = OptimizedHexAgent(Colour.RED, sims=50)
        move = agent.make_move(1, board, None)
        print(f"First move: {move}")
    except Exception as e:
        print(f"Agent test skipped (model not found): {e}")