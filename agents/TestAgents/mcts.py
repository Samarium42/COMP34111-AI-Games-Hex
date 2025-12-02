import math
from copy import deepcopy

from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        board: Board,
        player_to_move: Colour,
        parent: "MCTSNode | None" = None,
        move: tuple[int, int] | None = None,
    ):
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children: dict[tuple[int, int], "MCTSNode"] = {}
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = self._get_empty_tiles()

    def _get_empty_tiles(self) -> list[tuple[int, int]]:
        """Get all empty positions on the board."""
        empty = []
        for i in range(self.board.size):
            for j in range(self.board.size):
                if self.board.tiles[i][j].colour is None:
                    empty.append((i, j))
        return empty

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (someone won or board full)."""
        return self._check_winner() is not None or len(self._get_empty_tiles()) == 0

    def _check_winner(self) -> Colour | None:
        """Check if there's a winner. Returns the winning color or None."""
        # Check RED (top to bottom)
        if self._has_won(Colour.RED):
            return Colour.RED
        # Check BLUE (left to right)
        if self._has_won(Colour.BLUE):
            return Colour.BLUE
        return None

    def _has_won(self, colour: Colour) -> bool:
        """Check if the given color has won using DFS."""
        size = self.board.size
        visited = set()

        if colour == Colour.RED:
            # Start from top row, try to reach bottom row
            start_positions = [
                (0, j)
                for j in range(size)
                if self.board.tiles[0][j].colour == colour
            ]
            target_row = size - 1

            def is_goal(x, y):
                return x == target_row

        else:  # BLUE
            # Start from left column, try to reach right column
            start_positions = [
                (i, 0)
                for i in range(size)
                if self.board.tiles[i][0].colour == colour
            ]
            target_col = size - 1

            def is_goal(x, y):
                return y == target_col

        # DFS from each starting position
        for start in start_positions:
            if start in visited:
                continue
            stack = [start]
            while stack:
                x, y = stack.pop()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                if is_goal(x, y):
                    return True
                # Check neighbors
                for k in range(Tile.NEIGHBOUR_COUNT):
                    nx = x + Tile.I_DISPLACEMENTS[k]
                    ny = y + Tile.J_DISPLACEMENTS[k]
                    if 0 <= nx < size and 0 <= ny < size:
                        if (nx, ny) not in visited:
                            if self.board.tiles[nx][ny].colour == colour:
                                stack.append((nx, ny))
        return False

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> "MCTSNode":
        """Select the best child according to UCB1."""
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_constant),
        )

    def expand(self) -> "MCTSNode":
        """Expand by adding a new child node for an untried move."""
        move = self.untried_moves.pop()
        new_board = deepcopy(self.board)
        new_board.set_tile_colour(move[0], move[1], self.player_to_move)
        next_player = Colour.opposite(self.player_to_move)
        child = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children[move] = child
        return child