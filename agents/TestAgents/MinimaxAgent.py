"""
MinimaxAgent - Tactical Hex agent using Minimax with Alpha-Beta Pruning

USAGE:
------
Single game test:
    python Hex.py -p1 "agents.TestAgents.MinimaxAgent MinimaxAgent" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent" -v

Test against your main agent:
    python Hex.py -p1 "agents.TestAgents.MinimaxAgent MinimaxAgent" -p2 "agents.YourGroup.YourAgent YourAgent" -v

Tournament mode:
    Create cmd.txt with: agents.TestAgents.MinimaxAgent MinimaxAgent
    Then run: python HexTournament.py

STRATEGY:
---------
Uses Minimax algorithm with alpha-beta pruning for deep tactical search.
Searches 4-6 moves ahead with sophisticated board evaluation.
Strong at finding forced wins and blocking opponent threats.
"""

import sys
import numpy as np
from typing import List, Tuple, Optional, Dict
import time


from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MinimaxAgent(AgentBase):
    """
    Advanced tactical agent using Minimax with Alpha-Beta pruning.
    
    Features:
    - Depth-limited search (adjusts depth based on board fullness)
    - Alpha-beta pruning for efficiency
    - Sophisticated evaluation function
    - Move ordering for better pruning
    - Iterative deepening
    """
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._board_size = None
        self._my_colour = colour
        self._opp_colour = Colour.RED if colour == Colour.BLUE else Colour.BLUE
        self._time_limit = 170  # 170 seconds safety margin (out of 180s)
        self._start_time = None
        self._nodes_evaluated = 0
        
    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """Main decision function using iterative deepening minimax."""
        self._start_time = time.time()
        self._nodes_evaluated = 0
        
        # Initialize board size
        if self._board_size is None:
            self._board_size = len(board.tiles)
        
        # Turn 2: Evaluate swap
        if turn == 2:
            return self._should_swap(board, opp_move)
        
        # Check if game is nearly over (switch to deeper search)
        empty_count = self._count_empty(board)
        
        # Determine search depth based on game phase
        if empty_count > 80:  # Early game
            max_depth = 4
        elif empty_count > 40:  # Mid game
            max_depth = 5
        else:  # Late game
            max_depth = 6
        
        # Iterative deepening
        best_move = None
        for depth in range(1, max_depth + 1):
            if time.time() - self._start_time > self._time_limit:
                break
            
            move = self._find_best_move(board, depth)
            if move:
                best_move = move
        
        return best_move if best_move else self._get_fallback_move(board)
    
    def _should_swap(self, board: Board, opp_move: Optional[Move]) -> Move:
        """Evaluate whether to swap on turn 2."""
        if opp_move is None or opp_move.is_swap():
            return self._get_fallback_move(board)
        
        # Evaluate position from opponent's perspective
        score = self._evaluate_board(board, self._opp_colour)
        
        # Swap if opponent has significant advantage
        # Strong opening moves are typically in center region
        center = self._board_size // 2
        opp_x, opp_y = opp_move.x, opp_move.y
        
        # Center positions are strong - consider swapping
        distance_from_center = abs(opp_x - center) + abs(opp_y - center)
        
        if distance_from_center <= 2 and score > 1.0:
            return Move(-1, -1)  # Swap
        
        # Otherwise make our own move
        return self._find_best_move(board, depth=3)
    
    def _find_best_move(self, board: Board, depth: int) -> Optional[Move]:
        """Find best move using minimax with alpha-beta pruning."""
        empty_positions = self._get_empty_positions(board)
        
        if not empty_positions:
            return None
        
        # Order moves for better pruning (center positions first)
        ordered_moves = self._order_moves(empty_positions)
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for x, y in ordered_moves:
            if time.time() - self._start_time > self._time_limit:
                break
            
            # Make move
            board.set_tile_colour(x, y, self._my_colour)
            
            # Check immediate win
            if board.has_ended(self._my_colour):
                board.tiles[x][y]._colour = None
                return Move(x, y)
            
            # Minimax evaluation
            score = self._minimax(board, depth - 1, alpha, beta, False)
            
            # Undo move
            board.tiles[x][y]._colour = None
            
            if score > best_score:
                best_score = score
                best_move = Move(x, y)
            
            alpha = max(alpha, score)
        
        return best_move
    
    def _minimax(self, board: Board, depth: int, alpha: float, beta: float, 
                 is_maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Best score for maximizer
            beta: Best score for minimizer
            is_maximizing: True if maximizing player's turn
            
        Returns:
            Board evaluation score
        """
        self._nodes_evaluated += 1
        
        # Check time limit
        if time.time() - self._start_time > self._time_limit:
            return 0
        
        # Terminal conditions
        if depth == 0:
            return self._evaluate_board(board, self._my_colour)
        
        # Check for wins
        if board.has_ended(self._my_colour):
            return 10000 + depth  # Prefer faster wins
        if board.has_ended(self._opp_colour):
            return -10000 - depth  # Prefer slower losses
        
        empty_positions = self._get_empty_positions(board)
        if not empty_positions:
            return 0  # Draw (shouldn't happen in Hex)
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(empty_positions)
        
        if is_maximizing:
            max_eval = float('-inf')
            for x, y in ordered_moves:
                board.set_tile_colour(x, y, self._my_colour)
                eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.tiles[x][y]._colour = None
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval
        else:
            min_eval = float('inf')
            for x, y in ordered_moves:
                board.set_tile_colour(x, y, self._opp_colour)
                eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.tiles[x][y]._colour = None
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval
    
    def _evaluate_board(self, board: Board, colour: Colour) -> float:
        """
        Sophisticated board evaluation function.
        
        Considers:
        1. Connection strength (how close to winning)
        2. Piece mobility and influence
        3. Control of key regions
        4. Blocking opponent paths
        """
        my_col = colour
        opp_col = Colour.RED if colour == Colour.BLUE else Colour.BLUE
        
        # Calculate connection scores using path analysis
        my_connections = self._count_connections(board, my_col)
        opp_connections = self._count_connections(board, opp_col)
        
        # Strategic position value
        my_strategic = self._strategic_value(board, my_col)
        opp_strategic = self._strategic_value(board, opp_col)
        
        # Combine metrics
        score = (my_connections - opp_connections) * 10.0
        score += (my_strategic - opp_strategic) * 5.0
        
        return score
    
    def _count_connections(self, board: Board, colour: Colour) -> int:
        """Count strong connections toward goal."""
        size = self._board_size
        connection_count = 0
        
        if colour == Colour.RED:
            # RED: count pieces and influence toward bottom
            for x in range(size):
                for y in range(size):
                    if board.tiles[x][y].colour == colour:
                        # Value pieces closer to goal edge
                        connection_count += (size - x)
        else:  # BLUE
            # BLUE: count pieces and influence toward right
            for x in range(size):
                for y in range(size):
                    if board.tiles[x][y].colour == colour:
                        # Value pieces closer to goal edge
                        connection_count += (size - y)
        
        return connection_count
    
    def _strategic_value(self, board: Board, colour: Colour) -> int:
        """Calculate strategic position value (center control, key positions)."""
        size = self._board_size
        center = size // 2
        value = 0
        
        for x in range(size):
            for y in range(size):
                if board.tiles[x][y].colour == colour:
                    # Bonus for center control
                    distance_from_center = abs(x - center) + abs(y - center)
                    if distance_from_center <= 2:
                        value += 3
                    
                    # Bonus for having neighbors (strong positions)
                    neighbors = self._count_friendly_neighbors(board, x, y, colour)
                    value += neighbors
        
        return value
    
    def _count_friendly_neighbors(self, board: Board, x: int, y: int, 
                                   colour: Colour) -> int:
        """Count friendly neighbors for position."""
        count = 0
        for nx, ny in self._get_neighbors(x, y):
            if board.tiles[nx][ny].colour == colour:
                count += 1
        return count
    
    def _order_moves(self, positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Order moves for better alpha-beta pruning (center positions first)."""
        center = self._board_size // 2
        
        def move_priority(pos):
            x, y = pos
            # Prioritize center positions
            return -(abs(x - center) + abs(y - center))
        
        return sorted(positions, key=move_priority, reverse=True)
    
    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get valid hexagonal neighbors."""
        neighbors = []
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
    
    def _count_empty(self, board: Board) -> int:
        """Count empty tiles on board."""
        return len(self._get_empty_positions(board))
    
    def _get_fallback_move(self, board: Board) -> Move:
        """Fallback move if search fails (center or first available)."""
        center = self._board_size // 2
        
        # Try center
        if board.tiles[center][center].colour is None:
            return Move(center, center)
        
        # Try near center
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x, y = center + dx, center + dy
                if 0 <= x < self._board_size and 0 <= y < self._board_size:
                    if board.tiles[x][y].colour is None:
                        return Move(x, y)
        
        # First available
        empty = self._get_empty_positions(board)
        if empty:
            return Move(empty[0][0], empty[0][1])
        
        return Move(0, 0)