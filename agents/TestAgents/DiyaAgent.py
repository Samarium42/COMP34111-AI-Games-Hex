# agents/Group25/MCTSAgent.py
from __future__ import annotations

import time
import math
import random
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

from src.AgentBase import AgentBase
from src.Move import Move
from src.Colour import Colour


# =========================================================
#                 LIGHTWEIGHT BOARDSTATE
# =========================================================

class BoardState:
    EMPTY = 0
    RED = 1
    BLUE = 2

    NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    def __init__(self, size: int = 11):
        self.size = size
        self.grid: List[List[int]] = [[BoardState.EMPTY] * size for _ in range(size)]
        self.empty_cells: List[Tuple[int, int]] = [(x, y) for x in range(size) for y in range(size)]

        n = size * size
        self.parent_red = list(range(n + 2))
        self.parent_blue = list(range(n + 2))

        self.TOP = n
        self.BOTTOM = n + 1
        self.LEFT = n
        self.RIGHT = n + 1

        self._winner = None


    def _index(self, x: int, y: int) -> int:
        return x * self.size + y

    def _find(self, parent: List[int], a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(self, parent: List[int], a: int, b: int) -> None:
        ra = self._find(parent, a)
        rb = self._find(parent, b)
        if ra != rb:
            parent[rb] = ra

    def copy(self) -> "BoardState":
        new = BoardState(self.size)
        new.grid = [row[:] for row in self.grid]
        new.empty_cells = self.empty_cells[:]
        new.parent_red = self.parent_red[:]
        new.parent_blue = self.parent_blue[:]
        new._winner = self._winner
        return new

    def play(self, x: int, y: int, colour_int: int) -> None:
        if self.grid[x][y] != BoardState.EMPTY:
            raise ValueError("Invalid move - cell already occupied")

        self.grid[x][y] = colour_int

        try:
            self.empty_cells.remove((x, y))
        except ValueError:
            pass

        idx = self._index(x, y)

        if colour_int == BoardState.RED:
            parent = self.parent_red
            if x == 0:
                self._union(parent, idx, self.TOP)
            if x == self.size - 1:
                self._union(parent, idx, self.BOTTOM)
            for dx, dy in BoardState.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == BoardState.RED:
                        self._union(parent, idx, self._index(nx, ny))
            if self._find(parent, self.TOP) == self._find(parent, self.BOTTOM):
                self._winner = Colour.RED
        else:
            parent = self.parent_blue
            if y == 0:
                self._union(parent, idx, self.LEFT)
            if y == self.size - 1:
                self._union(parent, idx, self.RIGHT)
            for dx, dy in BoardState.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == BoardState.BLUE:
                        self._union(parent, idx, self._index(nx, ny))
            if self._find(parent, self.LEFT) == self._find(parent, self.RIGHT):
                self._winner = Colour.BLUE

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        return self.empty_cells

    def is_terminal(self) -> bool:
        return self._winner is not None

    def check_winner_fast(self):
        return self._winner

    @staticmethod
    def from_engine_board(board) -> "BoardState":
        size = board.size
        bs = BoardState(size)

        bs.empty_cells = []
        n = size * size
        bs.parent_red = list(range(n + 2))
        bs.parent_blue = list(range(n + 2))

        for x in range(size):
            for y in range(size):
                tile_colour = board.tiles[x][y].colour
                if tile_colour is None:
                    bs.grid[x][y] = BoardState.EMPTY
                    bs.empty_cells.append((x, y))
                elif tile_colour == Colour.RED:
                    bs.grid[x][y] = BoardState.RED
                else:
                    bs.grid[x][y] = BoardState.BLUE

        for x in range(size):
            for y in range(size):
                val = bs.grid[x][y]
                if val == BoardState.RED:
                    idx = bs._index(x, y)
                    if x == 0:
                        bs._union(bs.parent_red, idx, bs.TOP)
                    if x == size - 1:
                        bs._union(bs.parent_red, idx, bs.BOTTOM)
                    for dx, dy in BoardState.NEIGHBOURS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if bs.grid[nx][ny] == BoardState.RED:
                                bs._union(bs.parent_red, idx, bs._index(nx, ny))
                elif val == BoardState.BLUE:
                    idx = bs._index(x, y)
                    if y == 0:
                        bs._union(bs.parent_blue, idx, bs.LEFT)
                    if y == size - 1:
                        bs._union(bs.parent_blue, idx, bs.RIGHT)
                    for dx, dy in BoardState.NEIGHBOURS:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if bs.grid[nx][ny] == BoardState.BLUE:
                                bs._union(bs.parent_blue, idx, bs._index(nx, ny))

        if bs._find(bs.parent_red, bs.TOP) == bs._find(bs.parent_red, bs.BOTTOM):
            bs._winner = Colour.RED
        elif bs._find(bs.parent_blue, bs.LEFT) == bs._find(bs.parent_blue, bs.RIGHT):
            bs._winner = Colour.BLUE

        return bs


# =========================================================
#                    HELPER: HEURISTIC MOVE
# =========================================================

def pick_heuristic_move(state: BoardState, cur_int: int, opp_int: int) -> Tuple[int, int]:
    """
    Simple playout policy:
    - Prefer moves adjacent to our stones
    - Then moves adjacent to opponent stones (blocking)
    - Else random empty cell
    """
    size = state.size
    candidates_own: List[Tuple[int, int]] = []
    candidates_block: List[Tuple[int, int]] = []

    for (x, y) in state.get_empty_cells():
        own_adj = False
        opp_adj = False
        for dx, dy in BoardState.NEIGHBOURS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                val = state.grid[nx][ny]
                if val == cur_int:
                    own_adj = True
                elif val == opp_int:
                    opp_adj = True
        if own_adj:
            candidates_own.append((x, y))
        elif opp_adj:
            candidates_block.append((x, y))

    if candidates_own:
        return random.choice(candidates_own)
    if candidates_block:
        return random.choice(candidates_block)

    # fallback
    return random.choice(state.get_empty_cells())


# =========================================================
#                        MCTS NODE
# =========================================================

MAX_RAVE_MOVES = 20   # how many playout moves we use for RAVE updates
RAVE_BIAS = 300.0     # blending constant for RAVE


class Node:
    def __init__(
        self,
        state: BoardState,
        player_to_move: Colour,
        parent: "Node | None" = None,
        move: Move | None = None,
        root_player: Colour | None = None,
    ):
        self.state = state
        self.player_to_move = player_to_move
        self.parent = parent
        self.move = move
        self.root_player = root_player if root_player is not None else player_to_move

        self.children: List[Node] = []
        self.untried_moves: List[Tuple[int, int]] = state.get_empty_cells()[:]

        self.visits = 0
        self.wins = 0
        self.c = math.sqrt(2)

        # RAVE stats: move (x,y) -> wins/visits for the player_to_move at this node
        self.rave_wins: Dict[Tuple[int, int], float] = {}
        self.rave_visits: Dict[Tuple[int, int], float] = {}

    def uct_with_rave(self, child: "Node") -> float:
        if child.visits == 0:
            # totally unexplored child => try it
            return float("inf")

        key = (child.move.x, child.move.y)
        rv = self.rave_visits.get(key, 0.0)
        rw = self.rave_wins.get(key, 0.0)

        q = child.wins / child.visits if child.visits > 0 else 0.0
        q_rave = rw / rv if rv > 0 else 0.0

        # blend Q and RAVE
        beta = rv / (child.visits + rv + 1e-9 + 4.0 * rv * child.visits / RAVE_BIAS)
        blended = (1 - beta) * q + beta * q_rave

        # UCB exploration
        exploration = self.c * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-9))
        return blended + exploration

    def best_child(self) -> "Node":
        return max(self.children, key=lambda c: self.uct_with_rave(c))

    def select(self) -> "Node":
        node = self
        while not node.untried_moves and node.children:
            node = node.best_child()
        return node

    def expand(self) -> "Node":
        mx, my = self.untried_moves.pop()
        new_state = self.state.copy()

        cur_int = BoardState.RED if self.player_to_move == Colour.RED else BoardState.BLUE
        new_state.play(mx, my, cur_int)

        child = Node(
            state=new_state,
            player_to_move=Colour.opposite(self.player_to_move),
            parent=self,
            move=Move(mx, my),
            root_player=self.root_player,
        )
        self.children.append(child)
        return child

    def simulate(self) -> Tuple[Colour | None, List[Tuple[Colour, Tuple[int, int]]]]:
        state = self.state.copy()
        cur = self.player_to_move

        playout_moves: List[Tuple[Colour, Tuple[int, int]]] = []

        while True:
            winner = state.check_winner_fast()
            if winner is not None:
                return winner, playout_moves

            moves = state.get_empty_cells()
            if not moves:
                # board full; treat as loss for current player
                return Colour.opposite(cur), playout_moves

            cur_int = BoardState.RED if cur == Colour.RED else BoardState.BLUE
            opp_int = BoardState.BLUE if cur_int == BoardState.RED else BoardState.RED

            mx, my = pick_heuristic_move(state, cur_int, opp_int)
            state.play(mx, my, cur_int)
            playout_moves.append((cur, (mx, my)))

            cur = Colour.opposite(cur)

    # ---------- backpropagation + RAVE updates ----------

    def backpropagate(self, winner: Colour | None, playout_moves: List[Tuple[Colour, Tuple[int, int]]]) -> None:
        self.visits += 1
        if winner is not None and winner == self.root_player:
            self.wins += 1

        # RAVE / AMAF: treat playout moves as if they could have been played here
        # Only first MAX_RAVE_MOVES moves to control cost
        for idx, (player, (mx, my)) in enumerate(playout_moves):
            if idx >= MAX_RAVE_MOVES:
                break
            if player == self.player_to_move:
                key = (mx, my)
                self.rave_visits[key] = self.rave_visits.get(key, 0.0) + 1.0
                if winner == player:
                    self.rave_wins[key] = self.rave_wins.get(key, 0.0) + 1.0

        if self.parent:
            self.parent.backpropagate(winner, playout_moves)

    # ---------- run iterations (single tree) ----------

    def run_iterations(self, max_iterations: int, max_time: float | None) -> None:
        start = time.time()
        for _ in range(max_iterations):
            if max_time is not None and (time.time() - start) >= max_time:
                break

            node = self.select()
            if node.untried_moves:
                node = node.expand()
            winner, playout_moves = node.simulate()
            node.backpropagate(winner, playout_moves)


# =========================================================
#                         MCTS AGENT
# =========================================================

class DiyaAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.max_iterations = 20000   # total simulations per move (across all workers)
        self.max_time = 3.0           # hard time cap per move (seconds), or None
        self.debug = True             # print per-move stats
        self.num_workers = 2          # "root parallelisation" worker trees (uses threads)

    def _run_worker(self, root_state: BoardState, max_iterations: int, max_time: float | None):
        local_state = root_state.copy()
        root = Node(
            state=local_state,
            player_to_move=self.colour,
            parent=None,
            move=None,
            root_player=self.colour,
        )
        root.run_iterations(max_iterations=max_iterations, max_time=max_time)

        # return stats for children: (x,y) -> (visits, wins)
        stats: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for child in root.children:
            key = (child.move.x, child.move.y)
            stats[key] = (child.visits, child.wins)
        return stats

    def make_move(self, turn: int, board, opp_move) -> Move:
        root_state = BoardState.from_engine_board(board)

        # single-threaded case
        if self.num_workers <= 1:
            start = time.time()
            root = Node(
                state=root_state,
                player_to_move=self.colour,
                parent=None,
                move=None,
                root_player=self.colour,
            )
            root.run_iterations(self.max_iterations, self.max_time)
            elapsed = time.time() - start

            if not root.children:
                # fallback random
                mx, my = random.choice(root_state.get_empty_cells())
                if self.debug:
                    print(f"[MCTS] Iterations: 0, Time: {elapsed:.4f}s (fallback)")
                return Move(mx, my)

            best_child = max(root.children, key=lambda c: c.visits)
            if self.debug:
                print(f"[MCTS] Iterations: {self.max_iterations}, Time: {elapsed:.4f}s")
            return Move(best_child.move.x, best_child.move.y)

        # root-parallel using threads
        per_worker_iters = max(1, self.max_iterations // self.num_workers)
        start = time.time()
        aggregated: Dict[Tuple[int, int], List[float]] = {}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(
                    self._run_worker,
                    root_state,
                    per_worker_iters,
                    self.max_time,
                )
                for _ in range(self.num_workers)
            ]

            for fut in futures:
                worker_stats = fut.result()
                for key, (v, w) in worker_stats.items():
                    if key not in aggregated:
                        aggregated[key] = [0.0, 0.0]
                    aggregated[key][0] += v
                    aggregated[key][1] += w

        elapsed = time.time() - start
        total_iters = per_worker_iters * self.num_workers

        if not aggregated:
            mx, my = random.choice(root_state.get_empty_cells())
            if self.debug:
                print(f"[MCTS] Iterations: 0, Time: {elapsed:.4f}s (fallback parallel)")
            return Move(mx, my)

        # choose move with max aggregated visits
        best_move_xy = max(aggregated.items(), key=lambda item: item[1][0])[0]
        if self.debug:
            print(f"[MCTS] Iterations: {total_iters}, Time: {elapsed:.4f}s (workers={self.num_workers})")
        return Move(best_move_xy[0], best_move_xy[1])
