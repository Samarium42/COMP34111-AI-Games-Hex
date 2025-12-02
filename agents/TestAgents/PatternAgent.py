"""
PatternAgent - Tactical Hex agent using Pattern Recognition

USAGE:
------
Single game test:
    python Hex.py -p1 "agents.TestAgents.PatternAgent PatternAgent" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent" -v

Test against your main agent:
    python Hex.py -p1 "agents.TestAgents.PatternAgent PatternAgent" -p2 "agents.YourGroup.YourAgent YourAgent" -v

Tournament mode:
    Create cmd.txt with: agents.TestAgents.PatternAgent PatternAgent
    Then run: python HexTournament.py

STRATEGY:
---------
Recognizes critical Hex patterns (bridges, threats, edge templates).
Makes immediate tactical moves when patterns detected.
Falls back to heuristic search when no forcing patterns found.
Fast pattern matching for strong tactical play.
"""

import sys
import numpy as np
from typing import List, Tuple, Optional, Set
from collections import defaultdict

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move



class PatternAgent(AgentBase):
    """
    Advanced tactical agent using Hex pattern recognition.
    
    Recognizes:
    - 2-bridges (fundamental Hex connection pattern)
    - Edge templates (strong edge positions)
    - Threats (positions that create immediate winning threats)
    - Blocking patterns (critical defensive positions)
    - Forks (positions that create multiple threats)
    """
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._board_size = None
        self._my_colour = colour
        self._opp_colour = Colour.RED if colour == Colour.BLUE else Colour.BLUE
        
    def make_move(self, turn: int, board: Board, opp_move: Optional[Move]) -> Move:
        """Main decision function using pattern recognition."""
        # Initialize board size
        if self._board_size is None:
            self._board_size = len(board.tiles)
        
        # Turn 2: Evaluate swap
        if turn == 2:
            return self._should_swap(board, opp_move)
        
        # Priority 1: Check for immediate winning move
        winning_move = self._find_winning_move(board, self._my_colour)
        if winning_move:
            return winning_move
        
        # Priority 2: Block opponent's winning move
        blocking_move = self._find_winning_move(board, self._opp_colour)
        if blocking_move:
            return blocking_move
        
        # Priority 3: Look for bridge patterns to create
        bridge_move = self._find_bridge_move(board)
        if bridge_move:
            return bridge_move
        
        # Priority 4: Look for edge template positions
        edge_move = self._find_edge_template_move(board)
        if edge_move:
            return edge_move
        
        # Priority 5: Block opponent's bridges
        opp_bridge_block = self._find_opponent_bridge_block(board)
        if opp_bridge_block:
            return opp_bridge_block
        
        # Priority 6: Create threats (fork patterns)
        threat_move = self._find_threat_move(board)
        if threat_move:
            return threat_move
        
        # Fallback: Use heuristic evaluation
        return self._find_heuristic_move(board)
    
    def _should_swap(self, board: Board, opp_move: Optional[Move]) -> Move:
        """Evaluate whether to swap on turn 2."""
        if opp_move is None or opp_move.is_swap():
            return self._find_heuristic_move(board)
        
        # Strong center positions warrant swapping
        center = self._board_size // 2
        opp_x, opp_y = opp_move.x, opp_move.y
        
        # Calculate distance from center
        distance = abs(opp_x - center) + abs(opp_y - center)
        
        # Swap if opponent played in strong center region
        if distance <= 2:
            return Move(-1, -1)
        
        # Otherwise make our own move
        return self._find_heuristic_move(board)
    
    def _find_winning_move(self, board: Board, colour: Colour) -> Optional[Move]:
        """Find move that immediately wins the game."""
        empty_positions = self._get_empty_positions(board)
        
        for x, y in empty_positions:
            board.set_tile_colour(x, y, colour)
            if board.has_ended(colour):
                board.tiles[x][y]._colour = None
                return Move(x, y)
            board.tiles[x][y]._colour = None
        
        return None
    
    def _find_bridge_move(self, board: Board) -> Optional[Move]:
        """
        Find move that creates a 2-bridge pattern.
        
        A 2-bridge is two pieces with two shared empty neighbors.
        This guarantees connection in Hex.
        """
        my_pieces = self._get_colour_positions(board, self._my_colour)
        empty_positions = self._get_empty_positions(board)
        
        # For each empty position, check if it completes a bridge
        best_bridge_move = None
        best_bridge_score = 0
        
        for ex, ey in empty_positions:
            # Check if this position forms bridges with existing pieces
            bridge_count = self._count_bridges_formed(board, ex, ey, my_pieces)
            
            if bridge_count > best_bridge_score:
                best_bridge_score = bridge_count
                best_bridge_move = Move(ex, ey)
        
        return best_bridge_move if best_bridge_score > 0 else None
    
    def _count_bridges_formed(self, board: Board, x: int, y: int, 
                              my_pieces: List[Tuple[int, int]]) -> int:
        """Count how many bridges this move would form."""
        bridge_count = 0
        my_neighbors = []
        
        # Find friendly neighbors
        for nx, ny in self._get_neighbors(x, y):
            if (nx, ny) in my_pieces:
                my_neighbors.append((nx, ny))
        
        # For each pair of friendly neighbors, check if they form a bridge
        for i, (n1x, n1y) in enumerate(my_neighbors):
            for n2x, n2y in my_neighbors[i+1:]:
                if self._is_bridge_pattern(board, n1x, n1y, n2x, n2y, x, y):
                    bridge_count += 1
        
        return bridge_count
    
    def _is_bridge_pattern(self, board: Board, x1: int, y1: int, 
                          x2: int, y2: int, bridge_x: int, bridge_y: int) -> bool:
        """
        Check if three positions form a valid 2-bridge pattern.
        (x1,y1) and (x2,y2) are friendly pieces, (bridge_x, bridge_y) is candidate.
        """
        # Get common neighbors of the two pieces
        neighbors1 = set(self._get_neighbors(x1, y1))
        neighbors2 = set(self._get_neighbors(x2, y2))
        common = neighbors1 & neighbors2
        
        # Valid bridge: two empty positions that are common neighbors
        # One is our candidate, check if there's another empty one
        empty_common = [pos for pos in common 
                       if board.tiles[pos[0]][pos[1]].colour is None]
        
        return len(empty_common) == 2 and (bridge_x, bridge_y) in empty_common
    
    def _find_edge_template_move(self, board: Board) -> Optional[Move]:
        """
        Find move that secures edge template positions.
        Edge templates are strong patterns near edges.
        """
        size = self._board_size
        empty_positions = self._get_empty_positions(board)
        
        best_edge_move = None
        best_edge_score = 0
        
        for x, y in empty_positions:
            edge_score = self._evaluate_edge_position(board, x, y)
            if edge_score > best_edge_score:
                best_edge_score = edge_score
                best_edge_move = Move(x, y)
        
        return best_edge_move if best_edge_score > 5 else None
    
    def _evaluate_edge_position(self, board: Board, x: int, y: int) -> int:
        """Evaluate strategic value of edge position."""
        size = self._board_size
        score = 0
        
        # Check proximity to relevant edges
        if self._my_colour == Colour.RED:
            # RED wants to connect top-bottom
            if x <= 1:  # Near top edge
                score += 5
                # Bonus if has friendly neighbors
                for nx, ny in self._get_neighbors(x, y):
                    if board.tiles[nx][ny].colour == self._my_colour:
                        score += 2
            elif x >= size - 2:  # Near bottom edge
                score += 5
                for nx, ny in self._get_neighbors(x, y):
                    if board.tiles[nx][ny].colour == self._my_colour:
                        score += 2
        else:  # BLUE
            # BLUE wants to connect left-right
            if y <= 1:  # Near left edge
                score += 5
                for nx, ny in self._get_neighbors(x, y):
                    if board.tiles[nx][ny].colour == self._my_colour:
                        score += 2
            elif y >= size - 2:  # Near right edge
                score += 5
                for nx, ny in self._get_neighbors(x, y):
                    if board.tiles[nx][ny].colour == self._my_colour:
                        score += 2
        
        return score
    
    def _find_opponent_bridge_block(self, board: Board) -> Optional[Move]:
        """Find move that blocks opponent's bridge patterns."""
        opp_pieces = self._get_colour_positions(board, self._opp_colour)
        empty_positions = self._get_empty_positions(board)
        
        # Find opponent bridge points to block
        for ex, ey in empty_positions:
            bridge_count = self._count_bridges_formed(board, ex, ey, opp_pieces)
            if bridge_count > 0:
                return Move(ex, ey)
        
        return None
    
    def _find_threat_move(self, board: Board) -> Optional[Move]:
        """Find move that creates multiple threats (fork)."""
        empty_positions = self._get_empty_positions(board)
        
        best_threat_move = None
        best_threat_count = 0
        
        for x, y in empty_positions:
            # Simulate move
            board.set_tile_colour(x, y, self._my_colour)
            
            # Count how many winning moves this creates for next turn
            threat_count = 0
            for tx, ty in self._get_empty_positions(board):
                board.set_tile_colour(tx, ty, self._my_colour)
                if board.has_ended(self._my_colour):
                    threat_count += 1
                board.tiles[tx][ty]._colour = None
            
            board.tiles[x][y]._colour = None
            
            # Fork: creates 2+ winning threats
            if threat_count >= 2 and threat_count > best_threat_count:
                best_threat_count = threat_count
                best_threat_move = Move(x, y)
        
        return best_threat_move if best_threat_count >= 2 else None
    
    def _find_heuristic_move(self, board: Board) -> Move:
        """Fallback heuristic evaluation when no patterns found."""
        empty_positions = self._get_empty_positions(board)
        
        if not empty_positions:
            return Move(0, 0)
        
        best_move = None
        best_score = float('-inf')
        
        for x, y in empty_positions:
            score = self._evaluate_position(board, x, y)
            if score > best_score:
                best_score = score
                best_move = Move(x, y)
        
        return best_move if best_move else Move(empty_positions[0][0], empty_positions[0][1])
    
    def _evaluate_position(self, board: Board, x: int, y: int) -> float:
        """Heuristic evaluation of position value."""
        score = 0.0
        
        # Bonus for friendly neighbors (connectivity)
        for nx, ny in self._get_neighbors(x, y):
            if board.tiles[nx][ny].colour == self._my_colour:
                score += 3.0
            elif board.tiles[nx][ny].colour == self._opp_colour:
                score += 1.0  # Blocking is also valuable
        
        # Bonus for proximity to goal edge
        size = self._board_size
        if self._my_colour == Colour.RED:
            # Progress toward bottom
            progress = size - x
            score += progress * 0.5
        else:  # BLUE
            # Progress toward right
            progress = size - y
            score += progress * 0.5
        
        # Bonus for center positions (early game)
        center = size // 2
        center_distance = abs(x - center) + abs(y - center)
        score += (size - center_distance) * 0.2
        
        return score
    
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
    
    def _get_colour_positions(self, board: Board, colour: Colour) -> List[Tuple[int, int]]:
        """Get all positions of specified colour."""
        positions = []
        for x in range(self._board_size):
            for y in range(self._board_size):
                if board.tiles[x][y].colour == colour:
                    positions.append((x, y))
        return positions