"""
Group 3 - Rey Agent
MCTS-based Hex AI with Numba acceleration and parallel search
"""

import numpy as np
import random
import math
import time
from numba import njit
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from src.AgentBase import AgentBase
from src.Move import Move


# ============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# ============================================================================

@njit
def get_neighbors(x, y, size=11):
    """Get 6 hex neighbors"""
    neighbors = []
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors


@njit
def get_legal_moves(board, size=11):
    """Get all empty positions"""
    moves = []
    for y in range(size):
        for x in range(size):
            if board[y][x] == 0:
                moves.append((x, y))
    return moves


@njit
def check_win_red(board, size=11):
    """Check if Red won (top to bottom connection)"""
    visited = np.zeros((size, size), dtype=np.bool_)
    stack = []
    
    # Start from top row
    for x in range(size):
        if board[0][x] == 1:
            stack.append((x, 0))
            visited[0][x] = True
    
    while stack:
        x, y = stack.pop()
        if y == size - 1:
            return True
        
        for nx, ny in get_neighbors(x, y, size):
            if not visited[ny][nx] and board[ny][nx] == 1:
                visited[ny][nx] = True
                stack.append((nx, ny))
    return False


@njit
def check_win_blue(board, size=11):
    """Check if Blue won (left to right connection)"""
    visited = np.zeros((size, size), dtype=np.bool_)
    stack = []
    
    # Start from left column
    for y in range(size):
        if board[y][0] == 2:
            stack.append((0, y))
            visited[y][0] = True
    
    while stack:
        x, y = stack.pop()
        if x == size - 1:
            return True
        
        for nx, ny in get_neighbors(x, y, size):
            if not visited[ny][nx] and board[ny][nx] == 2:
                visited[ny][nx] = True
                stack.append((nx, ny))
    return False


@njit
def evaluate_move(board, x, y, color, size=11):
    """Heuristic evaluation of a move"""
    # Distance to goal edges
    if color == 1:  # Red - vertical
        goal_dist = min(y, size - 1 - y)
    else:  # Blue - horizontal
        goal_dist = min(x, size - 1 - x)
    
    # Count friendly neighbors (connectivity)
    connections = 0
    for nx, ny in get_neighbors(x, y, size):
        if board[ny][nx] == color:
            connections += 1
    
    # Centrality bonus
    center = size // 2
    centrality = -(abs(x - center) + abs(y - center)) * 0.5
    
    return -goal_dist * 2 + connections * 5 + centrality


@njit
def simulate_game(board, player, size=11):
    """
    Fast heuristic-guided rollout
    Returns winner: 1 (Red) or 2 (Blue)
    """
    sim = board.copy()
    current = player
    
    for _ in range(size * size):
        moves = get_legal_moves(sim, size)
        if len(moves) == 0:
            break
        
        # 85% greedy, 15% random
        if random.random() < 0.85 and len(moves) > 1:
            best_score = -99999
            best_move = moves[0]
            
            # Evaluate up to 15 moves
            sample_size = min(15, len(moves))
            for i in range(sample_size):
                move = moves[i]
                score = evaluate_move(sim, move[0], move[1], current, size)
                if score > best_score:
                    best_score = score
                    best_move = move
            
            x, y = best_move
        else:
            # Random exploration
            idx = random.randint(0, len(moves) - 1)
            x, y = moves[idx]
        
        sim[y][x] = current
        
        # Check terminal state
        if current == 1 and check_win_red(sim, size):
            return 1
        if current == 2 and check_win_blue(sim, size):
            return 2
        
        current = 3 - current
    
    # Fallback: count pieces
    return 1 if np.sum(sim == 1) >= np.sum(sim == 2) else 2


# ============================================================================
# THREAD-SAFE MCTS NODE
# ============================================================================

class ThreadSafeNode:
    """MCTS node with thread-safe operations for parallel search"""
    
    def __init__(self, board, parent=None, move=None, player=1):
        self.board = board
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        self.visits = 0
        self.wins = 0
        self.lock = Lock()
        self._untried = None
    
    @property
    def untried_moves(self):
        if self._untried is None:
            self._untried = get_legal_moves(self.board)
        return self._untried
    
    def ucb1(self, c=1.41):
        """Upper Confidence Bound formula"""
        with self.lock:
            if self.visits == 0:
                return float('inf')
            exploit = self.wins / self.visits
            explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
            return exploit + explore
    
    def select_child(self):
        """Select best child using UCB1"""
        return max(self.children, key=lambda n: n.ucb1())
    
    def expand(self, move, next_player):
        """Add a child node"""
        new_board = self.board.copy()
        # move is (x, y), board array is [y][x]
        new_board[move[1]][move[0]] = self.player
        child = ThreadSafeNode(new_board, self, move, next_player)
        
        with self.lock:
            self.children.append(child)
        return child
    
    def update(self, result):
        """Backpropagate result"""
        with self.lock:
            self.visits += 1
            self.wins += result


# ============================================================================
# REY AGENT - PARALLEL MCTS
# ============================================================================

class ReyAgent(AgentBase):
    """
    Group 3 Hex Agent
    
    Strategy:
    - Monte Carlo Tree Search with UCB1
    - Parallel root exploration (8 threads)
    - Heuristic-guided rollouts
    - Opening book for first moves
    """
    
    def __init__(self, color):
        super().__init__(color)
        self.size = 11
        # Colour enum: RED=0, BLUE=1, but we use 0 for empty
        # So our internal representation: 0=empty, 1=RED, 2=BLUE
        self.my_color = color.value + 1
        self.opp_color = 3 - self.my_color
        self.workers = 8
        print(f"ReyAgent initialized as {color.name}")
    
    def make_move(self, turn, board, last_move):
        """Main entry point called by game engine"""
        board_state = self._board_to_numpy(board)
        
        # Try opening book first
        opening = self._get_opening_move(board_state)
        if opening is not None:
            print(f"Opening book: {opening}")
            return Move(opening[0], opening[1])
        
        # Run parallel MCTS
        best_move = self._parallel_mcts(board_state, time_limit=3.5)
        return Move(best_move[0], best_move[1])
    
    def _board_to_numpy(self, board):
        """Convert Board object to numpy array"""
        arr = np.zeros((self.size, self.size), dtype=np.int32)
        
        tiles = board.tiles
        
        # CRITICAL FIX: Board.tiles is indexed as [x][y]
        # But we need our array as [y][x] for standard row/col access
        for x in range(self.size):
            for y in range(self.size):
                tile = tiles[x][y]
                if tile.colour is not None:
                    arr[y][x] = tile.colour.value + 1  # RED=1, BLUE=2
                else:
                    arr[y][x] = 0  # Empty
        
        return arr
    
    def _get_opening_move(self, board):
        """Strong opening book moves"""
        count = np.sum(board > 0)
        center = self.size // 2
        
        # First move: center is strongest
        if count == 0:
            return (center, center)
        
        # Second move: if opponent took center, play adjacent
        if count == 1 and board[center][center] != 0:
            good_moves = [
                (center - 1, center), (center + 1, center),
                (center, center - 1), (center, center + 1),
                (center - 1, center + 1), (center + 1, center - 1)
            ]
            for m in good_moves:
                if 0 <= m[0] < self.size and 0 <= m[1] < self.size:
                    if board[m[1]][m[0]] == 0:
                        return m
        
        return None
    
    def _parallel_mcts(self, board, time_limit):
        """Parallel MCTS with root parallelization"""
        root = ThreadSafeNode(board, player=self.opp_color)
        
        # Get legal moves
        moves = root.untried_moves
        if len(moves) == 0:
            return (self.size // 2, self.size // 2)
        
        # Evaluate and sort moves
        scored = []
        for m in moves:
            score = evaluate_move(board, m[0], m[1], self.my_color, self.size)
            scored.append((m, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Pre-expand root with top moves
        top_k = min(25, len(scored))
        for move, _ in scored[:top_k]:
            root.expand(move, self.my_color)
        
        # Parallel search
        iterations = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            while time.time() - start_time < time_limit:
                # Submit batch of iterations
                batch_size = self.workers * 4
                futures = [executor.submit(self._run_iteration, root) 
                          for _ in range(batch_size)]
                
                # Wait for batch to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                        iterations += 1
                    except Exception as e:
                        print(f"Iteration error: {e}")
        
        # Select best move by visit count
        best_child = max(root.children, key=lambda n: n.visits)
        win_rate = best_child.wins / best_child.visits if best_child.visits > 0 else 0
        
        print(f"ReyAgent MCTS: {iterations} sims | Move: {best_child.move} | "
              f"Visits: {best_child.visits} | WR: {win_rate:.1%}")
        
        return best_child.move
    
    def _run_iteration(self, root):
        """Single MCTS iteration: select, expand, simulate, backpropagate"""
        # Selection phase
        node = root
        path = [node]
        sim_board = root.board.copy()
        current_player = self.my_color
        
        while node.children:
            node = node.select_child()
            path.append(node)
            
            if node.move:
                # Update simulation board
                sim_board[node.move[1]][node.move[0]] = node.player
                current_player = 3 - node.player
        
        # Expansion phase
        moves = get_legal_moves(sim_board, self.size)
        if len(moves) > 0 and node.visits > 0:
            # Expand one child
            move = moves[random.randint(0, len(moves) - 1)]
            node = node.expand(move, current_player)
            path.append(node)
            sim_board[move[1]][move[0]] = current_player
            current_player = 3 - current_player
        
        # Simulation phase
        winner = simulate_game(sim_board, current_player, self.size)
        
        # Backpropagation phase
        result = 1 if winner == self.my_color else 0
        for n in path:
            n.update(result)