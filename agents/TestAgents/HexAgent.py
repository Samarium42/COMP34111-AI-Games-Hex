import time
from copy import deepcopy
from random import choice, random

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile

from agents.TestAgents.mcts import MCTSNode
from agents.TestAgents.heuristics import (
    should_swap,
    select_weighted_move,
    compute_board_hash,
)


class HexAgent(AgentBase):

    _board_size: int = 11
    _total_time: float = 300.0  # 5 minutes total
    _time_used: float = 0.0
    _move_count: int = 0

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._time_used = 0.0
        self._move_count = 0
        # Transposition table: hash -> (visits, wins)
        self._transposition_table: dict[int, tuple[int, float]] = {}

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a move based on the current board state."""
        start_time = time.time()

        # Handle swap decision on turn 2
        if turn == 2 and opp_move is not None:
            if self._should_swap(opp_move):
                return Move(-1, -1)

        move = self._select_move(board, turn)

        # Track time usage
        self._time_used += time.time() - start_time
        self._move_count += 1

        return move

    def _should_swap(self, opp_move: Move) -> bool:
        """Use heuristic-based swap decision."""
        return should_swap(opp_move.x, opp_move.y, self._board_size)

    def _get_time_allocation(self, turn: int, board: Board) -> float:
        """
        Dynamically allocate time based on game phase and remaining time.
        """
        remaining_time = self._total_time - self._time_used
        empty_count = sum(
            1 for i in range(board.size) for j in range(board.size)
            if board.tiles[i][j].colour is None
        )
        total_tiles = board.size * board.size

        # Estimate remaining moves (roughly half of empty tiles)
        estimated_moves_left = max(1, empty_count // 2)

        # Base allocation
        base_time = remaining_time / estimated_moves_left

        # Phase-based multiplier
        filled_ratio = 1 - (empty_count / total_tiles)

        if filled_ratio < 0.15:
            # Early game: allocate more time (opening is critical)
            multiplier = 1.5
        elif filled_ratio < 0.4:
            # Mid game: most critical decisions
            multiplier = 1.3
        elif filled_ratio < 0.7:
            # Late-mid game
            multiplier = 1.0
        else:
            # End game: positions are clearer, less time needed
            multiplier = 0.7

        # Emergency time management
        if remaining_time < 30:
            multiplier = 0.3
        elif remaining_time < 60:
            multiplier = 0.5

        allocated = min(base_time * multiplier, remaining_time * 0.15)
        return max(0.1, min(allocated, 10.0))  # Between 0.1s and 10s

    def _select_move(self, board: Board, turn: int) -> Move:
        """Use MCTS to select the best move with time management."""
        start_time = time.time()
        time_budget = self._get_time_allocation(turn, board)
        deadline = start_time + time_budget - 0.05  # Reserve 50ms

        # Create root node with current player
        root = MCTSNode(deepcopy(board), self._colour)
        root_hash = compute_board_hash(board)

        # If only one move available, return it immediately
        if len(root.untried_moves) == 1:
            move = root.untried_moves[0]
            return Move(move[0], move[1])

        # Check for immediate winning moves
        winning_move = self._find_winning_move(board)
        if winning_move:
            return Move(winning_move[0], winning_move[1])

        iterations = 0
        best_move_stable_count = 0
        last_best_move = None

        while time.time() < deadline:
            node = root

            # Selection: traverse tree using UCB1
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion: add a new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation: heuristic-guided playout
            winner = self._simulate_with_heuristics(node)

            # Backpropagation
            self._backpropagate(node, winner)

            iterations += 1

            # Early termination check (every 100 iterations)
            if iterations % 100 == 0 and root.children:
                current_best = max(
                    root.children.keys(),
                    key=lambda m: root.children[m].visits
                )
                if current_best == last_best_move:
                    best_move_stable_count += 1
                    # If best move is stable and has high win rate, can terminate early
                    if best_move_stable_count >= 5:
                        best_child = root.children[current_best]
                        if best_child.visits > 0:
                            win_rate = best_child.wins / best_child.visits
                            if win_rate > 0.85 or win_rate < 0.15:
                                break
                else:
                    best_move_stable_count = 0
                    last_best_move = current_best

        # Select best move (most visited child)
        if root.children:
            best_move = max(root.children.keys(), key=lambda m: root.children[m].visits)

            # Store in transposition table
            best_child = root.children[best_move]
            self._transposition_table[root_hash] = (best_child.visits, best_child.wins)

            return Move(best_move[0], best_move[1])

        # Fallback: use heuristic-guided selection
        empty = self._get_empty_tiles(board)
        if empty:
            move = select_weighted_move(empty, board, self._colour)
            if move:
                return Move(move[0], move[1])
            move = choice(empty)
            return Move(move[0], move[1])
        return Move(0, 0)  # Should never happen

    def _find_winning_move(self, board: Board) -> tuple[int, int] | None:
        """Check if there's an immediate winning move."""
        empty_tiles = self._get_empty_tiles(board)

        # Only check if few empty tiles (more likely to have winning move)
        if len(empty_tiles) > 30:
            return None

        for move in empty_tiles:
            # Make temporary move
            board.set_tile_colour(move[0], move[1], self._colour)

            # Check if we won
            if self._check_win(board, self._colour):
                # Undo move
                board.set_tile_colour(move[0], move[1], None)
                return move

            # Undo move
            board.set_tile_colour(move[0], move[1], None)

        return None

    def _simulate_with_heuristics(self, node: MCTSNode) -> Colour | None:
        """Simulate with heuristic-guided move selection."""
        board = deepcopy(node.board)
        current_player = node.player_to_move
        empty_tiles = [
            (i, j)
            for i in range(board.size)
            for j in range(board.size)
            if board.tiles[i][j].colour is None
        ]

        move_count = 0
        while empty_tiles:
            # Check for winner periodically
            if move_count % 5 == 0 or len(empty_tiles) < 10:
                winner = self._check_winner_fast(board)
                if winner is not None:
                    return winner

            # Use heuristic-guided selection with some probability
            if len(empty_tiles) > 5 and random() < 0.6:
                move = select_weighted_move(empty_tiles, board, current_player)
            else:
                # Pure random for speed in late game
                idx = choice(range(len(empty_tiles)))
                move = empty_tiles[idx]

            empty_tiles.remove(move)
            board.set_tile_colour(move[0], move[1], current_player)
            current_player = Colour.opposite(current_player)
            move_count += 1

        return self._check_winner_fast(board)

    def _simulate(self, node: MCTSNode) -> Colour | None:
        """Simulate a random game from the given node."""
        board = deepcopy(node.board)
        current_player = node.player_to_move
        empty_tiles = [
            (i, j)
            for i in range(board.size)
            for j in range(board.size)
            if board.tiles[i][j].colour is None
        ]

        # Random playout
        while empty_tiles:
            # Check for winner periodically (every 5 moves for speed)
            if len(empty_tiles) % 5 == 0 or len(empty_tiles) < 10:
                winner = self._check_winner_fast(board)
                if winner is not None:
                    return winner

            idx = choice(range(len(empty_tiles)))
            move = empty_tiles.pop(idx)
            board.set_tile_colour(move[0], move[1], current_player)
            current_player = Colour.opposite(current_player)

        # Final winner check
        return self._check_winner_fast(board)

    def _check_winner_fast(self, board: Board) -> Colour | None:
        """Fast winner check using DFS."""
        size = board.size
        # Check RED
        visited = set()
        for j in range(size):
            if board.tiles[0][j].colour == Colour.RED and (0, j) not in visited:
                if self._dfs_win(board, 0, j, Colour.RED, visited, size):
                    return Colour.RED
        # Check BLUE
        visited = set()
        for i in range(size):
            if board.tiles[i][0].colour == Colour.BLUE and (i, 0) not in visited:
                if self._dfs_win(board, i, 0, Colour.BLUE, visited, size):
                    return Colour.BLUE
        return None

    def _dfs_win(
        self,
        board: Board,
        x: int,
        y: int,
        colour: Colour,
        visited: set,
        size: int,
    ) -> bool:
        """DFS to check if color has won."""
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            # Goal check
            if colour == Colour.RED and cx == size - 1:
                return True
            if colour == Colour.BLUE and cy == size - 1:
                return True
            # Neighbors
            for k in range(Tile.NEIGHBOUR_COUNT):
                nx = cx + Tile.I_DISPLACEMENTS[k]
                ny = cy + Tile.J_DISPLACEMENTS[k]
                if 0 <= nx < size and 0 <= ny < size:
                    if (nx, ny) not in visited and board.tiles[nx][ny].colour == colour:
                        stack.append((nx, ny))
        return False

    def _backpropagate(self, node: MCTSNode, winner: Colour | None) -> None:
        """Backpropagate the result up the tree."""
        while node is not None:
            node.visits += 1
            if winner is not None:
                # The node stores player_to_move, so the PARENT made the move
                # We reward the parent if the winner matches who made the move
                if node.parent is not None:
                    parent_player = node.parent.player_to_move
                    if winner == parent_player:
                        node.wins += 1
                else:
                    # Root node
                    if winner == self._colour:
                        node.wins += 1
            node = node.parent

    def _get_empty_tiles(self, board: Board) -> list[tuple[int, int]]:
        """Get all empty tile positions on the board."""
        empty = []
        for i in range(self._board_size):
            for j in range(self._board_size):
                if board.tiles[i][j].colour is None:
                    empty.append((i, j))
        return empty

    def _check_win(self, board: Board, colour: Colour) -> bool:
        """Check if the given colour has won."""
        visited = set()
        size = board.size
        if colour == Colour.RED:
            for j in range(size):
                if board.tiles[0][j].colour == colour and (0, j) not in visited:
                    if self._dfs_win(board, 0, j, colour, visited, size):
                        return True
        else:
            for i in range(size):
                if board.tiles[i][0].colour == colour and (i, 0) not in visited:
                    if self._dfs_win(board, i, 0, colour, visited, size):
                        return True
        return False