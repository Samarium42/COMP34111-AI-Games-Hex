import math
import random
import time
from dataclasses import dataclass, field

from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour
from src.Tile import Tile


NEIGHBOUR_OFFSETS = list(zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS))


@dataclass(slots=True)
class MCTSNode:
    """
    One node in the MCTS search tree.
    """
    state: list
    parent: "MCTSNode | None" = None
    move_from_parent: tuple | None = None
    player_to_move: int = 1      # +1 = RED, -1 = BLUE (the *actual* player)
    wins: float = 0.0
    visits: int = 0
    children: list = field(default_factory=list)
    untried_moves: list = field(default_factory=list)


class MCTSHexAgent(AgentBase):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size: int | None = None
        self.max_move_time: float = 0.6
        self.total_time_used: float = 0.0

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================
    def make_move(self, turn: int, board: Board, previous_move: Move) -> Move:
        start_time = time.time()
        self.board_size = board.size

        root_state = self._board_to_array(board)
        # Root player is the actual colour we control: RED=+1, BLUE=-1
        root_player = 1 if self.colour == Colour.RED else -1

        # --------------------------------------------------------------
        # Turn-2 swap decision (only allowed for BLUE).
        # --------------------------------------------------------------
        if turn == 2 and self.colour == Colour.BLUE:
            no_swap_wr = self.estimate_winrate(
                state=root_state,
                player=root_player,
                time_budget=0.2,
            )

            # Hypothetical swap: we become RED
            swap_wr = self.estimate_winrate(
                state=root_state,
                player=+1,
                time_budget=0.2,
            )

            if swap_wr > no_swap_wr:
                # Perform swap in the environment
                self.colour = Colour.RED
                self.total_time_used += time.time() - start_time
                return Move(-1, -1)

        # --------------------------------------------------------------
        # Normal MCTS for move selection
        # --------------------------------------------------------------
        root = MCTSNode(
            state=root_state,
            parent=None,
            move_from_parent=None,
            player_to_move=root_player,
        )
        root.untried_moves = self._legal_moves(root_state)

        end_time = start_time + self.max_move_time
        self._run_mcts(root, root_player, end_time)

        self.total_time_used += time.time() - start_time

        # Choose move: most visited child, fallback to random legal move
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            move_choice = best_child.move_from_parent
        else:
            move_choice = random.choice(root.untried_moves)

        x, y = move_choice
        return Move(x, y)

    # ==================================================================
    # TOP-LEVEL WIN RATE ESTIMATOR (USED FOR SWAP DECISION)
    # ==================================================================
    def estimate_winrate(self, state, player: int, time_budget: float = 0.2) -> float:
        """
        Run a short MCTS search from (state, player) and return an
        approximate win probability for 'player'.
        """
        root = MCTSNode(
            state=[row[:] for row in state],
            parent=None,
            move_from_parent=None,
            player_to_move=player,
        )
        root.untried_moves = self._legal_moves(root.state)

        end_time = time.time() + time_budget
        self._run_mcts(root, player, end_time)

        if root.visits == 0:
            return 0.5
        return root.wins / root.visits

    # ==================================================================
    # CORE MCTS LOOP
    # ==================================================================
    def _run_mcts(self, root: MCTSNode, root_player: int, end_time: float) -> None:
        """
        Run MCTS starting from 'root' until 'end_time' (wall-clock).
        """
        while time.time() < end_time:
            # 1. Selection + Expansion
            node, rollout_state, next_player = self._tree_policy(root)

            # 2. Simulation
            result = self._default_policy(rollout_state, next_player)

            # 3. Backpropagation
            self._backup(node, result, root_player)

    # ------------------------------------------------------------------
    # TREE POLICY: SELECTION + EXPANSION
    # ------------------------------------------------------------------
    def _tree_policy(self, root: MCTSNode):
        """
        Starting from 'root', descend using UCT until we find a node
        that can be expanded or is terminal. Returns:
          (node_reached, state_at_node, player_to_move_at_node)
        """
        node = root
        # Work on a copy of the board state
        state = [row[:] for row in root.state]
        player = root.player_to_move

        # Selection: follow UCT while the node is fully expanded
        while not node.untried_moves and node.children:
            node = self._uct_select_child(node)
            move = node.move_from_parent
            state = self._apply_move(state, move, player)
            player = -player

        # Expansion: if we still have possible moves, expand one
        if node.untried_moves:
            move = node.untried_moves.pop(
                random.randrange(len(node.untried_moves))
            )
            state = self._apply_move(state, move, player)
            child = MCTSNode(
                state=state,
                parent=node,
                move_from_parent=move,
                player_to_move=-player,
            )
            child.untried_moves = self._legal_moves(state)
            node.children.append(child)
            node = child
            player = -player

        return node, state, player

    # ------------------------------------------------------------------
    # DEFAULT POLICY: RANDOM ROLLOUT
    # ------------------------------------------------------------------
    def _default_policy(self, state, player: int) -> int:
        """
        Play random moves from 'state' until a winner is found or
        the board is full. Returns +1 (RED), -1 (BLUE) or 0.
        """
        current_state = [row[:] for row in state]
        moves = self._legal_moves(current_state)
        current_player = player

        # Shuffle once instead of repeatedly picking random moves
        random.shuffle(moves)

        for move in moves:
            x, y = move
            current_state[x][y] = current_player

            winner = self._check_winner(current_state)
            if winner != 0:
                return winner

            current_player = -current_player

        return 0

    # ------------------------------------------------------------------
    # BACKUP: PROPAGATE RESULT UP THE TREE
    # ------------------------------------------------------------------
    def _backup(self, node: MCTSNode, result: int, root_player: int) -> None:
        """
        Propagate simulation result back up from 'node' to the root.
        Reward is 1.0 if the outcome equals root_player, else 0.0.
        """
        while node is not None:
            node.visits += 1
            if result == root_player:
                node.wins += 1.0
            node = node.parent

    # ==================================================================
    # BOARD / MOVE UTILITIES
    # ==================================================================
    def _board_to_array(self, board: Board):
        """
        Convert the game's Board object into a simple integer grid.

        IMPORTANT: This encodes the *actual* colours:
            RED  -> +1
            BLUE -> -1
            empty -> 0
        """
        size = board.size
        arr = [[0] * size for _ in range(size)]

        for x in range(size):
            for y in range(size):
                tile_colour = board.tiles[x][y].colour
                if tile_colour is None:
                    arr[x][y] = 0
                elif tile_colour == Colour.RED:
                    arr[x][y] = 1
                else:  # Colour.BLUE
                    arr[x][y] = -1

        return arr

    def _legal_moves(self, state):
        """
        Return list of coordinates (x, y) where the board is empty.
        """
        size = len(state)
        moves = []
        for x in range(size):
            row = state[x]
            for y in range(size):
                if row[y] == 0:
                    moves.append((x, y))
        return moves

    def _apply_move(self, state, move, player: int):
        """
        Return a *new* state with 'move' applied by 'player'.
        """
        x, y = move
        new_state = [row[:] for row in state]
        new_state[x][y] = player
        return new_state

    # ==================================================================
    # WIN CHECKING (HEX CONNECTIVITY)
    # ==================================================================
    def _check_winner(self, state) -> int:
        """
        Check if RED (+1) or BLUE (-1) has a connecting path.
        Returns: +1 if RED wins, -1 if BLUE wins, 0 otherwise.
        """
        size = len(state)

        # ------------- RED: top to bottom (value +1) -------------
        visited_red = [[False] * size for _ in range(size)]

        def dfs_red(i, j) -> bool:
            if j == size - 1:
                return True
            visited_red[i][j] = True
            for di, dj in NEIGHBOUR_OFFSETS:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    if not visited_red[ni][nj] and state[ni][nj] == 1:
                        if dfs_red(ni, nj):
                            return True
            return False

        for i in range(size):
            if state[i][0] == 1 and dfs_red(i, 0):
                return 1

        # ------------- BLUE: left to right (value -1) -------------
        visited_blue = [[False] * size for _ in range(size)]

        def dfs_blue(i, j) -> bool:
            if i == size - 1:
                return True
            visited_blue[i][j] = True
            for di, dj in NEIGHBOUR_OFFSETS:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    if not visited_blue[ni][nj] and state[ni][nj] == -1:
                        if dfs_blue(ni, nj):
                            return True
            return False

        for j in range(size):
            if state[0][j] == -1 and dfs_blue(0, j):
                return -1

        return 0

    # ==================================================================
    # UCT CHILD SELECTION
    # ==================================================================
    def _uct_select_child(self, node: MCTSNode, exploration_constant: float = 1.4) -> MCTSNode:
        """
        Select a child of 'node' using the UCT formula.
        """
        # Add 1 to avoid log(0)
        log_parent_visits = math.log(node.visits + 1e-9)
        best_score = float("-inf")
        best_child = None

        for child in node.children:
            # Small epsilon in denominator to avoid division by zero
            inv_sqrt_visits = 1.0 / math.sqrt(child.visits + 1e-9)
            exploit = child.wins / (child.visits + 1e-9)
            explore = exploration_constant * math.sqrt(log_parent_visits) * inv_sqrt_visits
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child
