import math
import random
import time
from src.AgentBase import AgentBase
from src.Board import Board
from src.Move import Move
from src.Colour import Colour
from src.Tile import Tile


NEIGHBOUR_OFFSETS = list(zip(Tile.I_DISPLACEMENTS, Tile.J_DISPLACEMENTS))


class Node:
    __slots__ = (
        "parent",
        "children",
        "move_from_parent",
        "wins",
        "visits",
        "untried_moves",
        "player_to_move",
        "state",
    )

    def __init__(self, state, parent=None, move_from_parent=None, player_to_move=1):
        self.state = state
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = []
        self.player_to_move = player_to_move


class MCTSAgent(AgentBase):

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.board_size = None
        self.max_move_time = 0.6
        self.total_time_used = 0.0

    # ********************************************************************
    # MAKE MOVE (NOW WITH MCTS-BASED SWAP DECISION)
    # ********************************************************************
    def make_move(self, turn: int, board: Board, previous_move: Move) -> Move:
        start_time = time.time()
        self.board_size = board.size

        root_state = self._board_to_array(board)
        root_player = 1 if self.colour == Colour.RED else -1

        # --------------------------------------------------------------
        # OPTION A: TRUE MCTS-BASED SWAP DECISION
        # Only for BLUE on turn 2
        # --------------------------------------------------------------
        if turn == 2 and self.colour == Colour.BLUE:
            # Evaluate no-swap position
            no_swap_wr = self.estimate_winrate(
                state=root_state,
                player=root_player,
                time_budget=0.2
            )

            # Evaluate swap position (as if we swapped and became RED)
            swapped_state = root_state     # same stones
            swapped_player = +1            # after swap, WE are RED
            swap_wr = self.estimate_winrate(
                state=swapped_state,
                player=swapped_player,
                time_budget=0.2
            )

            # Choose better
            if swap_wr > no_swap_wr:
                # Perform swap
                self.colour = Colour.RED
                return Move(-1, -1)

        # --------------------------------------------------------------
        # Otherwise: normal MCTS to choose move
        # --------------------------------------------------------------
        root = Node(root_state, parent=None, player_to_move=root_player)
        root.untried_moves = self.get_legal_moves(root_state)

        end_time = start_time + self.max_move_time

        while time.time() < end_time:
            node = root
            state = [row[:] for row in root_state]
            player = root_player

            # Selection
            while not node.untried_moves and node.children:
                node = self.uct_select_child(node)
                move = node.move_from_parent
                state = self.apply_move(state, move, player)
                player = -player

            # Expansion
            if node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                move = node.untried_moves.pop(idx)
                state = self.apply_move(state, move, player)
                child = Node(state, parent=node,
                             move_from_parent=move,
                             player_to_move=-player)
                child.untried_moves = self.get_legal_moves(state)
                node.children.append(child)
                node = child
                player = -player

            # Simulation
            result = self.random_rollout(state, player)

            # Backpropagation
            while node is not None:
                node.visits += 1
                reward = 1.0 if result == root_player else 0.0
                node.wins += reward
                node = node.parent

        self.total_time_used += time.time() - start_time

        # Best child = highest visits
        if not root.children:
            move_choice = random.choice(root.untried_moves)
        else:
            move_choice = max(root.children, key=lambda c: c.visits).move_from_parent

        x, y = move_choice
        return Move(x, y)

    # ********************************************************************
    # ESTIMATE WIN RATE (NEW!)
    # ********************************************************************
    def estimate_winrate(self, state, player, time_budget=0.2):
        root = Node(state, parent=None, player_to_move=player)
        root.untried_moves = self.get_legal_moves(state)

        end_time = time.time() + time_budget

        while time.time() < end_time:
            node = root
            sim_state = [row[:] for row in state]
            sim_player = player

            # Selection
            while not node.untried_moves and node.children:
                node = self.uct_select_child(node)
                move = node.move_from_parent
                sim_state = self.apply_move(sim_state, move, sim_player)
                sim_player = -sim_player

            # Expansion
            if node.untried_moves:
                idx = random.randrange(len(node.untried_moves))
                move = node.untried_moves.pop(idx)
                sim_state = self.apply_move(sim_state, move, sim_player)
                child = Node(sim_state, parent=node,
                             move_from_parent=move,
                             player_to_move=-sim_player)
                child.untried_moves = self.get_legal_moves(sim_state)
                node.children.append(child)
                node = child
                sim_player = -sim_player

            # Simulation
            result = self.random_rollout(sim_state, sim_player)

            # Backprop
            while node is not None:
                node.visits += 1
                reward = 1.0 if result == player else 0.0
                node.wins += reward
                node = node.parent

        if root.visits == 0:
            return 0.5
        return root.wins / root.visits

    # ********************************************************************
    # BOARD / MOVE UTILITIES
    # ********************************************************************
    def _board_to_array(self, board: Board):
        size = board.size
        arr = [[0] * size for _ in range(size)]
        for x in range(size):
            for y in range(size):
                tile_colour = board.tiles[x][y].colour
                if tile_colour is None:
                    arr[x][y] = 0
                elif tile_colour == self.colour:
                    arr[x][y] = 1
                else:
                    arr[x][y] = -1
        return arr

    def get_legal_moves(self, state):
        moves = []
        size = len(state)
        for x in range(size):
            for y in range(size):
                if state[x][y] == 0:
                    moves.append((x, y))
        return moves

    def apply_move(self, state, move, player):
        s2 = [row[:] for row in state]
        x, y = move
        s2[x][y] = player
        return s2

    # ********************************************************************
    # ROLLOUT AND WIN CHECKING
    # ********************************************************************
    def random_rollout(self, state, player):
        current_state = [row[:] for row in state]
        moves = self.get_legal_moves(current_state)
        current_player = player

        while moves:
            move = random.choice(moves)
            current_state[move[0]][move[1]] = current_player

            winner = self.check_winner(current_state)
            if winner != 0:
                return winner

            current_player = -current_player
            moves.remove(move)

        return 0

    def check_winner(self, state):
        size = len(state)

        # RED top→bottom (value +1)
        visited = [[False] * size for _ in range(size)]

        def dfs_red(x, y):
            if y == size - 1:
                return True
            visited[x][y] = True
            for dx, dy in NEIGHBOUR_OFFSETS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if not visited[nx][ny] and state[nx][ny] == 1:
                        if dfs_red(nx, ny):
                            return True
            return False

        for x in range(size):
            if state[x][0] == 1 and dfs_red(x, 0):
                return 1

        # BLUE left→right (value -1)
        visited = [[False] * size for _ in range(size)]

        def dfs_blue(x, y):
            if x == size - 1:
                return True
            visited[x][y] = True
            for dx, dy in NEIGHBOUR_OFFSETS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    if not visited[nx][ny] and state[nx][ny] == -1:
                        if dfs_blue(nx, ny):
                            return True
            return False

        for y in range(size):
            if state[0][y] == -1 and dfs_blue(0, y):
                return -1

        return 0

    # ********************************************************************
    # UCT
    # ********************************************************************
    def uct_select_child(self, node, C=1.4):
        log_total = math.log(node.visits + 1)
        best_score, best_child = -1e9, None
        for c in node.children:
            exploit = c.wins / (c.visits + 1e-9)
            explore = C * math.sqrt(log_total / (c.visits + 1e-9))
            score = exploit + explore
            if score > best_score:
                best_score, best_child = score, c
        return best_child
