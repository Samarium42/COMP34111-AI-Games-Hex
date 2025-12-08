import time
from math import inf
from typing import List, Tuple

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile
from src.Game import logger


class MinimaxBeamAgent(AgentBase):
    """
    A Hex-playing agent based on minimax search with alpha–beta pruning
    and beam search.

    Key ideas
    ---------
    - Uses an evaluation combining:
        * Approximate shortest connection distance (main term).
        * Stone counts.
        * Connectivity / bridge patterns.
    - Uses beam search:
        * At each node, only the top `beam_width` moves (by heuristic ordering)
          are explored deeply.
    - Uses iterative deepening:
        * Searches depth 1, then 2, ... up to `max_depth`, respecting a
          per-move time limit.
    """

    def __init__(
        self,
        colour: Colour,
        max_depth: int = 3,
        beam_width: int = 10,
        time_limit_seconds: float = 1.5,
    ):
        """
        Initialise the MinimaxBeamAgent.

        Parameters
        ----------
        colour : Colour
            The colour assigned to this agent (Colour.RED or Colour.BLUE).
        max_depth : int
            Maximum search depth for minimax.
        beam_width : int
            Maximum number of moves to keep after heuristic ordering at
            each node. If <= 0, beam search is effectively disabled.
        time_limit_seconds : float
            Per-move time budget. The search stops early if this is exceeded.
        """
        super().__init__(colour)
        self._max_depth = max_depth
        self._beam_width = beam_width      # <= 0 => approximate full minimax
        self._time_limit = time_limit_seconds
        self._start_time = 0.0

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Main entry point called by the game engine to obtain the agent's move.

        Implements:
        - Swap decision on turn 2 (pie rule).
        - Iterative deepening minimax with beam search and alpha–beta pruning.
        - Time control via `_time_up`.

        Parameters
        ----------
        turn : int
            Current turn number (1-based).
        board : Board
            Current board state (deep-copied by the engine).
        opp_move : Move | None
            Opponent's previous move. None on the very first move.

        Returns
        -------
        Move
            The chosen move for this turn. May be (-1, -1) for swap on turn 2.
        """
        # Record the time at which this move started, for per-move time limit.
        self._start_time = time.perf_counter()

        # Generate all legal moves (empty cells).
        legal_moves = self._generate_legal_moves(board)

        # Handle pie rule: if we are Player 2 (turn == 2) and the opponent
        # has just played their first move, decide whether to swap.
        if turn == 2 and opp_move is not None:
            if self._should_swap(board, opp_move):
                # Swap move is represented as (-1, -1).
                return Move(-1, -1)

        # No legal moves is extremely unlikely, but handle safely.
        if not legal_moves:
            logger.warning("No legal moves; returning (0,0) as fallback.")
            return Move(0, 0)

        # Only one legal move: no need to search.
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Default best move and value before search.
        best_move = legal_moves[0]
        best_value = -inf

        # Iterative deepening: increase depth from 1 up to max_depth.
        for depth in range(1, self._max_depth + 1):
            if self._time_up():
                break

            value, move = self._search_root(board, depth)

            # Stop if time is up after completing this depth.
            if self._time_up():
                break

            # Update best move if a better one was found at this depth.
            if move is not None:
                best_move = move
                best_value = value

            logger.debug(
                f"[MinimaxBeamAgent] depth={depth}, best_value={best_value}, "
                f"best_move=({best_move.x},{best_move.y})"
            )

        return best_move

    def _should_swap(self, board: Board, opp_move: Move) -> bool:
        """
        Decide whether to take the swap (pie rule) on turn 2.

        Current policy:
        - Compute Manhattan distance of opponent's first move from the centre.
        - If very central (dist <= 1), treat as strong and swap.
        - If far (dist >= 4), treat as weak and do not swap.
        - Middle band: currently defaults to no swap.

        This is a simple heuristic and can be tuned further.

        Parameters
        ----------
        board : Board
            Current board state.
        opp_move : Move
            Opponent's first move.

        Returns
        -------
        bool
            True if agent should swap, False otherwise.
        """
        size = board.size
        center = size // 2

        # Distance of their first stone from the centre of the board.
        dist = abs(opp_move.x - center) + abs(opp_move.y - center)

        # Very naive policy: strong, near-centre opening => swap.
        if dist <= 1:
            return True   # strong opening, grab it
        if dist >= 4:
            return False  # weak far-edge opening
        # Intermediate distance: future improvement could simulate both scenarios.
        return False

    # ------------------------------------------------------------------
    # Root search for our colour
    # ------------------------------------------------------------------
    def _search_root(self, board: Board, depth: int) -> Tuple[float, Move | None]:
        """
        Perform one layer of minimax search at the root for the given depth.

        - Generates legal moves.
        - Orders them using `_order_moves`.
        - Applies beam search (keep only top-k moves).
        - Performs alpha–beta minimax on child nodes.

        Parameters
        ----------
        board : Board
            Current board state.
        depth : int
            The depth limit for this search iteration.

        Returns
        -------
        (float, Move | None)
            The best evaluation value found and the corresponding move.
        """
        legal_moves = self._generate_legal_moves(board)

        # Heuristic move ordering for our colour.
        ordered_moves = self._order_moves(board, legal_moves, self.colour)

        # Beam search: restrict to top `beam_width` moves if applicable.
        if self._beam_width > 0 and len(ordered_moves) > self._beam_width:
            ordered_moves = ordered_moves[: self._beam_width]

        best_val = -inf
        best_move: Move | None = None
        alpha, beta = -inf, inf

        # Explore each candidate move using minimax.
        for move in ordered_moves:
            if self._time_up():
                break

            # Apply our move and let minimax evaluate the resulting position.
            child = self._apply_move(board, move, self.colour)
            val = self._minimax(
                child,
                depth - 1,
                player_to_move=self.opp_colour(),
                alpha=alpha,
                beta=beta,
            )

            # Track the best move and value so far.
            if val > best_val or best_move is None:
                best_val = val
                best_move = move

            # Standard alpha–beta update.
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break  # pruning

        return best_val, best_move

    def _count_stones(self, board: Board, colour: Colour) -> int:
        """
        Count how many stones of a given colour are on the board.

        Parameters
        ----------
        board : Board
            Current board state.
        colour : Colour
            The colour to count.

        Returns
        -------
        int
            Number of stones of the given colour.
        """
        size = board.size
        tiles = board.tiles
        c = 0
        for x in range(size):
            for y in range(size):
                if tiles[x][y].colour == colour:
                    c += 1
        return c

    def _connectivity_score(self, board: Board, colour: Colour) -> int:
        """
        Compute a simple connectivity score for a given colour.

        Components:
        - +1 for each stone in the interior (not on outermost rows/columns).
        - +1 per "bridge-like" local pattern around the stone.

        Parameters
        ----------
        board : Board
            Current board state.
        colour : Colour
            Colour whose connectivity is evaluated.

        Returns
        -------
        int
            Connectivity score (higher is better).
        """
        size = board.size
        tiles = board.tiles
        score = 0

        for x in range(size):
            for y in range(size):
                if tiles[x][y].colour == colour:
                    # Central bonus: interior positions are typically more valuable.
                    if 1 <= x <= size - 2 and 1 <= y <= size - 2:
                        score += 1

                    # Add local bridge pattern score.
                    score += self._count_bridges_around(board, x, y, colour)

        return score

    def _count_bridges_around(self, board: Board, x: int, y: int, colour: Colour) -> int:
        """
        Count possible "bridge-like" patterns around a given stone.

        A bridge here is approximated by:
        - A pair of neighbouring positions that are either:
          * Friendly stones, or
          * Empty cells where future friendly stones could go.

        Parameters
        ----------
        board : Board
            Current board state.
        x, y : int
            Coordinates of the reference stone.
        colour : Colour
            Colour of the reference stone.

        Returns
        -------
        int
            Number of bridge-like patterns detected around (x, y).
        """
        size = board.size
        tiles = board.tiles
        bridges = 0

        # Simple patterns approximating common Hex bridge shapes.
        patterns = [
            [(x, y - 1), (x, y + 1)],
            [(x - 1, y), (x + 1, y)],
            [(x - 1, y + 1), (x + 1, y - 1)],
        ]

        for pattern in patterns:
            empty_count = 0
            friendly_count = 0
            for px, py in pattern:
                if 0 <= px < size and 0 <= py < size:
                    c = tiles[px][py].colour
                    if c == colour:
                        friendly_count += 1
                    elif c is None:
                        empty_count += 1
            # Require at least one friendly stone and at least one potential extension.
            if friendly_count >= 1 and empty_count >= 1:
                bridges += 1

        return bridges

    # ------------------------------------------------------------------
    # Recursive minimax + alpha–beta + beam
    # ------------------------------------------------------------------
    def _minimax(
        self,
        board: Board,
        depth: int,
        player_to_move: Colour,
        alpha: float,
        beta: float,
    ) -> float:
        """
        Recursive minimax search with alpha–beta pruning and beam search.

        Parameters
        ----------
        board : Board
            Current board state.
        depth : int
            Remaining depth to search.
        player_to_move : Colour
            Colour whose turn it is at this node.
        alpha : float
            Current alpha bound for pruning (best guaranteed MAX value).
        beta : float
            Current beta bound for pruning (best guaranteed MIN value).

        Returns
        -------
        float
            Evaluation of this node from our agent's perspective.
        """
        # If we run out of time, return a static evaluation.
        if self._time_up():
            return self._evaluate(board)

        # Check terminal state (win/loss).
        winner = self._get_winner_safe(board)
        if winner is not None:
            return inf if winner == self.colour else -inf

        # If depth limit reached, evaluate statically.
        if depth == 0:
            return self._evaluate(board)

        # Generate legal moves. If none, evaluate statically.
        legal_moves = self._generate_legal_moves(board)
        if not legal_moves:
            return self._evaluate(board)

        # Order moves heuristically for the current player.
        ordered_moves = self._order_moves(board, legal_moves, player_to_move)

        # Beam search: only keep the top `beam_width` moves.
        if self._beam_width > 0 and len(ordered_moves) > self._beam_width:
            ordered_moves = ordered_moves[: self._beam_width]

        # MAX node: our turn
        if player_to_move == self.colour:
            value = -inf
            for move in ordered_moves:
                child = self._apply_move(board, move, player_to_move)
                child_val = self._minimax(
                    child,
                    depth - 1,
                    player_to_move=self.opp_colour(),
                    alpha=alpha,
                    beta=beta,
                )
                value = max(value, child_val)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break  # alpha–beta cut-off
            return value
        # MIN node: opponent's turn
        else:
            value = inf
            for move in ordered_moves:
                child = self._apply_move(board, move, player_to_move)
                child_val = self._minimax(
                    child,
                    depth - 1,
                    player_to_move=self.colour,
                    alpha=alpha,
                    beta=beta,
                )
                value = min(value, child_val)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # alpha–beta cut-off
            return value

    # ------------------------------------------------------------------
    # Evaluation: distance + material + connectivity
    # ------------------------------------------------------------------
    def _evaluate(self, board: Board) -> float:
        """
        Static evaluation of a board position from our perspective.

        Components
        ----------
        1. Connection distance (dominant term):
           - Approximate minimal "cost" to build a winning connection
             (our colour vs opponent).
           - Cost per empty cell = 1, friendly cell = 0, enemy cell = large.
           - Score term: opp_dist - my_dist.
        2. Material:
           - Difference in number of stones (our stones - opponent stones).
        3. Connectivity:
           - Difference in connectivity scores (bridges, centrality).

        Returns
        -------
        float
            Heuristic value; larger is better for us.
        """
        me = self.colour
        opp = self.opp_colour()

        # 1. Connection distance (biggest factor)
        my_dist = self._shortest_connection_distance(board, me)
        opp_dist = self._shortest_connection_distance(board, opp)

        if my_dist is None:
            my_dist = 1e6
        if opp_dist is None:
            opp_dist = 1e6

        conn_score = opp_dist - my_dist  # larger is better for us

        # 2. Stone counts (material)
        my_stones = self._count_stones(board, me)
        opp_stones = self._count_stones(board, opp)
        material_score = my_stones - opp_stones

        # 3. Bridges / connectivity
        my_conn = self._connectivity_score(board, me)
        opp_conn = self._connectivity_score(board, opp)
        connectivity_score = my_conn - opp_conn

        # Weighted combination of components.
        return (
            8.0 * conn_score +      # connection potential
            1.0 * material_score +  # slight preference for more stones
            3.0 * connectivity_score
        )

    # ------------------------------------------------------------------
    # Board helpers
    # ------------------------------------------------------------------
    def _generate_legal_moves(self, board: Board) -> List[Move]:
        """
        Generate all legal moves for the given board.

        A legal move is any empty tile.

        Parameters
        ----------
        board : Board
            Board state used to detect empties.

        Returns
        -------
        List[Move]
            List of all moves where the tile is empty.
        """
        moves: List[Move] = []
        size = board.size
        tiles = board.tiles

        for x in range(size):
            for y in range(size):
                if tiles[x][y].colour is None:
                    moves.append(Move(x, y))
        return moves

    def _apply_move(self, board: Board, move: Move, colour: Colour) -> Board:
        """
        Apply a move on a fresh copy of the board.

        The original board given by the engine must not be modified.

        Parameters
        ----------
        board : Board
            Original board state.
        move : Move
            The move to apply (x, y coordinates).
        colour : Colour
            Colour to place at (x, y).

        Returns
        -------
        Board
            New Board instance with the move applied.
        """
        size = board.size
        new_board = Board(board_size=size)

        # Copy all stone colours across.
        for x in range(size):
            for y in range(size):
                new_board.tiles[x][y].colour = board.tiles[x][y].colour

        # Place the new stone.
        new_board.set_tile_colour(move.x, move.y, colour)
        return new_board

    def _get_winner_safe(self, board: Board) -> Colour | None:
        """
        Safely check if the game has ended and return the winning colour.

        Wraps Board.has_ended / get_winner in a try/except to ensure that
        any unexpected errors do not crash the agent.

        Parameters
        ----------
        board : Board
            Board to be checked.

        Returns
        -------
        Colour | None
            Winning colour (Colour.RED or Colour.BLUE), or None if no winner.
        """
        try:
            if board.has_ended(Colour.RED):
                return board.get_winner()
            if board.has_ended(Colour.BLUE):
                return board.get_winner()
            return None
        except Exception as e:
            logger.error(f"[MinimaxBeamAgent] Exception in _get_winner_safe: {e}")
            return None

    # ------------------------------------------------------------------
    # Move ordering (crucial for pruning & beam search)
    # ------------------------------------------------------------------
    def _order_moves(
        self, board: Board, moves: List[Move], player_to_move: Colour
    ) -> List[Move]:
        """
        Order legal moves using a heuristic combining:
        - Centrality: moves closer to the centre of the board are favoured.
        - Adjacency: moves adjacent to existing stones of the player are favoured.
        - Static evaluation: evaluation of the child position.

        The resulting scores are used for:
        - Alpha–beta pruning effectiveness (good moves first).
        - Beam search (keeping only the top-k moves).

        Parameters
        ----------
        board : Board
            Current board state.
        moves : List[Move]
            List of candidate moves.
        player_to_move : Colour
            Colour about to move.

        Returns
        -------
        List[Move]
            Moves sorted from most promising to least.
        """
        size = board.size
        center = size // 2
        tiles = board.tiles

        scored = []
        for move in moves:
            cx, cy = move.x, move.y

            # Centrality: negative distance from centre (larger is better).
            centrality = -(abs(cx - center) + abs(cy - center))

            # Adjacency: count friendly neighbours.
            adj = 0
            for k in range(Tile.NEIGHBOUR_COUNT):
                nx = cx + Tile.I_DISPLACEMENTS[k]
                ny = cy + Tile.J_DISPLACEMENTS[k]
                if 0 <= nx < size and 0 <= ny < size:
                    if tiles[nx][ny].colour == player_to_move:
                        adj += 1

            # Static priority from centrality and adjacency.
            static_score = 2 * centrality + 3 * adj

            # Evaluate the resulting position after making this move.
            child = self._apply_move(board, move, player_to_move)
            eval_score = self._evaluate(child)

            # Combined heuristic score for ordering.
            scored.append((eval_score + static_score, move))

        # Sort moves by descending heuristic score.
        scored.sort(key=lambda p: p[0], reverse=True)
        return [m for (_, m) in scored]

    # ------------------------------------------------------------------
    # Shortest connection distance (Dijkstra on hex graph)
    # ------------------------------------------------------------------
    def _shortest_connection_distance(
        self, board: Board, colour: Colour
    ) -> float | None:
        """
        Estimate how "far" a player is from completing a winning connection.

        Uses a Dijkstra-like algorithm on the hex graph, where:
        - Own stones have cost 0.
        - Empty cells have cost 1.
        - Opponent stones have very high cost (treated as almost blocked).

        For RED:
            Connects top row (x=0) to bottom row (x=size-1).
        For BLUE:
            Connects left column (y=0) to right column (y=size-1).

        Parameters
        ----------
        board : Board
            Current board.
        colour : Colour
            Colour whose connection distance is evaluated.

        Returns
        -------
        float | None
            Estimated connection cost (smaller is better), or None if unreachable.
        """
        import heapq

        size = board.size
        tiles = board.tiles
        INF_DIST = 10**6

        # dist[x][y] = best known cost to reach cell (x, y)
        dist = [[INF_DIST] * size for _ in range(size)]
        pq: List[tuple[float, int, int]] = []

        di = Tile.I_DISPLACEMENTS
        dj = Tile.J_DISPLACEMENTS

        # Initialise frontier based on which sides this colour connects.
        if colour == Colour.RED:
            # RED: connect top to bottom, start from top row (x = 0).
            for y in range(size):
                cost = self._cell_cost(tiles[0][y].colour, colour)
                dist[0][y] = cost
                heapq.heappush(pq, (cost, 0, y))
            target = "BOTTOM"
        else:  # Colour.BLUE
            # BLUE: connect left to right, start from left column (y = 0).
            for x in range(size):
                cost = self._cell_cost(tiles[x][0].colour, colour)
                dist[x][0] = cost
                heapq.heappush(pq, (cost, x, 0))
            target = "RIGHT"

        # Dijkstra search.
        while pq:
            cur_dist, x, y = heapq.heappop(pq)
            if cur_dist != dist[x][y]:
                continue  # skip outdated entries

            # Check if target side reached.
            if target == "BOTTOM" and x == size - 1:
                return cur_dist
            if target == "RIGHT" and y == size - 1:
                return cur_dist

            # Relax edges to neighbours.
            for k in range(Tile.NEIGHBOUR_COUNT):
                nx = x + di[k]
                ny = y + dj[k]
                if 0 <= nx < size and 0 <= ny < size:
                    step_cost = self._cell_cost(tiles[nx][ny].colour, colour)
                    nd = cur_dist + step_cost
                    if nd < dist[nx][ny]:
                        dist[nx][ny] = nd
                        heapq.heappush(pq, (nd, nx, ny))

        # If no path found, return a very large cost.
        return float(INF_DIST)

    def _cell_cost(self, cell_colour: Colour | None, player_colour: Colour) -> float:
        """
        Cost of stepping onto a given cell during the distance computation.

        - Own stone: 0 (already part of the connection).
        - Empty cell: 1 (we might need to play there).
        - Opponent stone: large (treated as strongly blocking).

        Parameters
        ----------
        cell_colour : Colour | None
            Colour on this cell (or None if empty).
        player_colour : Colour
            Colour for which we are computing the cost.

        Returns
        -------
        float
            Cost value to step on this cell.
        """
        if cell_colour is None:
            return 1.0
        if cell_colour == player_colour:
            return 0.0
        return 1000.0

    # ------------------------------------------------------------------
    # Time control
    # ------------------------------------------------------------------
    def _time_up(self) -> bool:
        """
        Check whether the agent's per-move time budget has been exceeded.

        Returns
        -------
        bool
            True if elapsed time >= `_time_limit`, False otherwise.
        """
        return (time.perf_counter() - self._start_time) >= self._time_limit
