"""
DijkstraAgent - Strategic Hex agent using shortest path algorithms

USAGE:
------
Single game test:
    python Hex.py -p1 "agents.TestAgents.DijkstraAgent DijkstraAgent" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent" -v

Test against your main agent:
    python Hex.py -p1 "agents.TestAgents.DijkstraAgent DijkstraAgent" -p2 "agents.YourGroup.YourAgent YourAgent" -v

Tournament mode:
    Create cmd.txt in agents/TestAgents/ with: agents.TestAgents.DijkstraAgent DijkstraAgent
    Then run: python HexTournament.py

STRATEGY:
---------
Uses Dijkstra's algorithm to evaluate board positions and find optimal moves.
Minimizes own path to goal while maximizing opponent's path distance.
"""

import sys
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Optional

# Add parent directory to path to import game modules
import numpy as np
from heapq import heappush, heappop
from typing import List, Tuple, Optional

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

class DijkstraAgent(AgentBase):
    """
    Advanced strategic agent using Dijkstra's shortest path algorithm.
    
    Strategy:
    1. For each possible move, calculate shortest path distances for both players
    2. Evaluate position based on: (opponent_distance - my_distance)
    3. Choose move that maximizes this score
    4. Handles swap decision by comparing initial position value
    """
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._board_size = None
        self._my_colour = colour
        self._opp_colour = Colour.RED if colour == Colour.BLUE else Colour.BLUE
        
    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """
        Main decision function. Evaluates all possible moves and returns the best.
        
        Args:
            turn: Current turn number
            board: Current board state (deep copy)
            opp_move: Opponent's last move (None on turn 1)
            
        Returns:
            Move object with best position or swap
        """
        # Initialize board size on first call
        if self._board_size is None:
            self._board_size = len(board.tiles)
        
        # Turn 2 decision: evaluate whether to swap
        if turn == 2:
            return self._should_swap(board, opp_move)
        
        # Find all empty positions
        empty_positions = self._get_empty_positions(board)
        
        if not empty_positions:
            # Should never happen in valid game, but safety fallback
            return Move(0, 0)
        
        # Evaluate all possible moves
        best_move = None
        best_score = float('-inf')
        
        for x, y in empty_positions:
            score = self._evaluate_move(board, x, y)
            if score > best_score:
                best_score = score
                best_move = Move(x, y)
        
        return best_move
    
    def _should_swap(self, board: Board, opp_move: Optional[Move]) -> Move:
        """
        Decide whether to swap on turn 2.
        Swaps if opponent's opening move gives them advantage.
        
        Returns:
            Move(-1, -1) for swap, or a regular move
        """
        if opp_move is None or opp_move.is_swap():
            # Shouldn't happen, but play normally
            return self._find_best_first_move(board)
        
        # Evaluate current position (if we don't swap)
        current_score = self._evaluate_position(board)
        
        # Swap is good if opponent has advantage (negative score for us means opponent ahead)
        # Use threshold: swap if opponent's move is significantly strong
        if current_score < -0.5:  # Opponent has meaningful advantage
            return Move(-1, -1)
        
        # Otherwise, make our own move
        return self._find_best_first_move(board)
    
    def _find_best_first_move(self, board: Board) -> Move:
        """Find best move for turn 1 or when choosing not to swap."""
        empty_positions = self._get_empty_positions(board)
        
        best_move = None
        best_score = float('-inf')
        
        for x, y in empty_positions:
            score = self._evaluate_move(board, x, y)
            if score > best_score:
                best_score = score
                best_move = Move(x, y)
        
        return best_move if best_move else Move(self._board_size // 2, self._board_size // 2)
    
    def _evaluate_move(self, board: Board, x: int, y: int) -> float:
        """
        Evaluate a potential move at position (x, y).
        
        Strategy:
        - Simulate placing our piece at (x, y)
        - Calculate shortest paths for both players
        - Return score = opp_distance - my_distance (higher is better)
        
        Returns:
            Float score (higher = better move)
        """
        # Simulate move
        original_colour = board.tiles[x][y].colour
        board.set_tile_colour(x, y, self._my_colour)
        
        # Calculate distances
        my_distance = self._shortest_path_distance(board, self._my_colour)
        opp_distance = self._shortest_path_distance(board, self._opp_colour)
        
        # Restore board
        board.tiles[x][y]._colour = original_colour
        
        # Score: we want opponent far, us close
        # Add small bonus for center positions (tie-breaker)
        center = self._board_size // 2
        center_bonus = -0.01 * (abs(x - center) + abs(y - center))
        
        return (opp_distance - my_distance) + center_bonus
    
    def _evaluate_position(self, board: Board) -> float:
        """Evaluate current board position without making a move."""
        my_distance = self._shortest_path_distance(board, self._my_colour)
        opp_distance = self._shortest_path_distance(board, self._opp_colour)
        return opp_distance - my_distance
    
    def _shortest_path_distance(self, board: Board, colour: Colour) -> float:
        """
        Calculate shortest path from start edge to goal edge using Dijkstra's algorithm.
        
        For RED: start = top row (x=0), goal = bottom row (x=size-1)
        For BLUE: start = left column (y=0), goal = right column (y=size-1)
        
        Returns:
            Float distance (lower = closer to winning)
            Returns infinity if no path exists
        """
        size = self._board_size
        
        # Priority queue: (distance, x, y)
        pq = []
        distances = np.full((size, size), float('inf'))
        
        # Initialize start positions
        if colour == Colour.RED:
            # RED starts from top row (x=0)
            for y in range(size):
                if board.tiles[0][y].colour == colour:
                    distances[0][y] = 0
                    heappush(pq, (0, 0, y))
                elif board.tiles[0][y].colour is None:
                    distances[0][y] = 1
                    heappush(pq, (1, 0, y))
        else:  # BLUE
            # BLUE starts from left column (y=0)
            for x in range(size):
                if board.tiles[x][0].colour == colour:
                    distances[x][0] = 0
                    heappush(pq, (0, x, 0))
                elif board.tiles[x][0].colour is None:
                    distances[x][0] = 1
                    heappush(pq, (1, x, 0))
        
        # Dijkstra's algorithm
        while pq:
            dist, x, y = heappop(pq)
            
            # Skip if we've found a better path already
            if dist > distances[x][y]:
                continue
            
            # Check if we reached goal
            if colour == Colour.RED and x == size - 1:
                return dist
            elif colour == Colour.BLUE and y == size - 1:
                return dist
            
            # Explore neighbors
            for nx, ny in self._get_neighbors(x, y):
                tile = board.tiles[nx][ny]
                
                # Calculate edge weight
                if tile.colour == colour:
                    weight = 0  # Our pieces are free
                elif tile.colour is None:
                    weight = 1  # Empty tiles cost 1
                else:
                    weight = float('inf')  # Opponent pieces are impassable
                
                new_dist = dist + weight
                
                if new_dist < distances[nx][ny]:
                    distances[nx][ny] = new_dist
                    heappush(pq, (new_dist, nx, ny))
        
        # No path found (shouldn't happen in normal play)
        return float('inf')
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid hexagonal neighbors for position (x, y).
        Uses the same neighbor logic as Tile class.
        """
        neighbors = []
        # Hexagonal neighbor offsets (from Tile.py)
        i_displacements = [-1, -1, 0, 1, 1, 0]
        j_displacements = [0, 1, 1, 0, -1, -1]
        
        for di, dj in zip(i_displacements, j_displacements):
            nx, ny = x + di, y + dj
            if 0 <= nx < self._board_size and 0 <= ny < self._board_size:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def _get_empty_positions(self, board: Board) -> List[Tuple[int, int]]:
        """Get all empty board positions."""
        empty = []
        for x in range(self._board_size):
            for y in range(self._board_size):
                if board.tiles[x][y].colour is None:
                    empty.append((x, y))
        return empty