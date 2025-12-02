"""Hex-specific heuristics for MCTS enhancement."""

import heapq
import random as _random

from src.Board import Board
from src.Colour import Colour
from src.Tile import Tile


# Two-bridge patterns: relative offsets that form virtual connections
# A two-bridge exists when two stones of same color can be connected
# via two empty cells that share neighbors with both stones
TWO_BRIDGE_PATTERNS = [
    # (dx1, dy1, dx2, dy2, empty1, empty2) - stone offsets and empty cell offsets
    ((1, 0), (0, 1), (1, 1)),     # Bridge pattern 1
    ((-1, 1), (0, 1), (-1, 2)),   # Bridge pattern 2
    ((-1, 0), (-1, 1), (-2, 1)),  # Bridge pattern 3
    ((1, -1), (0, -1), (1, -2)),  # Bridge pattern 4
    ((1, 0), (1, -1), (2, -1)),   # Bridge pattern 5
    ((-1, 0), (0, -1), (-1, -1)), # Bridge pattern 6
]


def get_neighbors(x: int, y: int, size: int) -> list[tuple[int, int]]:
    """Get valid neighboring positions."""
    neighbors = []
    for k in range(Tile.NEIGHBOUR_COUNT):
        nx = x + Tile.I_DISPLACEMENTS[k]
        ny = y + Tile.J_DISPLACEMENTS[k]
        if 0 <= nx < size and 0 <= ny < size:
            neighbors.append((nx, ny))
    return neighbors


def detect_virtual_connections(
    board: Board, colour: Colour
) -> list[tuple[int, int]]:
    """
    Detect moves that complete virtual connections (two-bridges).
    Returns list of moves that would complete a two-bridge pattern.
    """
    size = board.size
    completing_moves = []

    for x in range(size):
        for y in range(size):
            if board.tiles[x][y].colour != colour:
                continue

            # Check each two-bridge pattern
            for pattern in TWO_BRIDGE_PATTERNS:
                (dx1, dy1), (dx2, dy2), (ex, ey) = pattern

                # Position of the other stone in the bridge
                x2, y2 = x + dx1 + dx2, y + dy1 + dy2

                if not (0 <= x2 < size and 0 <= y2 < size):
                    continue
                if board.tiles[x2][y2].colour != colour:
                    continue

                # Check the two empty cells that form the bridge
                e1x, e1y = x + dx1, y + dy1
                e2x, e2y = x + ex, y + ey

                if not (0 <= e1x < size and 0 <= e1y < size):
                    continue
                if not (0 <= e2x < size and 0 <= e2y < size):
                    continue

                e1_empty = board.tiles[e1x][e1y].colour is None
                e2_empty = board.tiles[e2x][e2y].colour is None

                # If one is occupied by opponent, the completing move is important
                if e1_empty and not e2_empty:
                    if board.tiles[e2x][e2y].colour == Colour.opposite(colour):
                        completing_moves.append((e1x, e1y))
                elif e2_empty and not e1_empty:
                    if board.tiles[e1x][e1y].colour == Colour.opposite(colour):
                        completing_moves.append((e2x, e2y))

    return completing_moves


def shortest_path_distance(
    board: Board, colour: Colour
) -> int:
    """
    Calculate shortest path distance to victory using Dijkstra.
    Empty cells cost 1, own cells cost 0, opponent cells cost infinity.
    Returns distance (lower is better for the player).
    """
    size = board.size
    INF = size * size + 1

    # For RED: top to bottom (row 0 to row size-1)
    # For BLUE: left to right (col 0 to col size-1)

    dist = [[INF] * size for _ in range(size)]
    heap = []

    if colour == Colour.RED:
        # Start from top row
        for j in range(size):
            tile_colour = board.tiles[0][j].colour
            if tile_colour == Colour.opposite(colour):
                continue
            cost = 0 if tile_colour == colour else 1
            dist[0][j] = cost
            heapq.heappush(heap, (cost, 0, j))
    else:  # BLUE
        # Start from left column
        for i in range(size):
            tile_colour = board.tiles[i][0].colour
            if tile_colour == Colour.opposite(colour):
                continue
            cost = 0 if tile_colour == colour else 1
            dist[i][0] = cost
            heapq.heappush(heap, (cost, i, 0))

    while heap:
        d, x, y = heapq.heappop(heap)

        if d > dist[x][y]:
            continue

        # Check goal
        if colour == Colour.RED and x == size - 1:
            return d
        if colour == Colour.BLUE and y == size - 1:
            return d

        for nx, ny in get_neighbors(x, y, size):
            tile_colour = board.tiles[nx][ny].colour
            if tile_colour == Colour.opposite(colour):
                continue

            edge_cost = 0 if tile_colour == colour else 1
            new_dist = d + edge_cost

            if new_dist < dist[nx][ny]:
                dist[nx][ny] = new_dist
                heapq.heappush(heap, (new_dist, nx, ny))

    # Find minimum distance to goal edge
    min_dist = INF
    if colour == Colour.RED:
        for j in range(size):
            min_dist = min(min_dist, dist[size - 1][j])
    else:
        for i in range(size):
            min_dist = min(min_dist, dist[i][size - 1])

    return min_dist


def evaluate_position(board: Board, colour: Colour) -> float:
    """
    Evaluate board position for a player.
    Returns positive score if position is good for colour.
    """
    my_dist = shortest_path_distance(board, colour)
    opp_dist = shortest_path_distance(board, Colour.opposite(colour))

    # Lower distance is better, so we want opp_dist - my_dist
    return opp_dist - my_dist


def get_move_priority(
    board: Board,
    move: tuple[int, int],
    colour: Colour,
) -> float:
    """
    Calculate priority score for a move during simulation.
    Higher score = more likely to be selected.
    """
    x, y = move
    size = board.size
    score = 0.0

    # Bonus for extending existing connections
    for nx, ny in get_neighbors(x, y, size):
        if board.tiles[nx][ny].colour == colour:
            score += 2.0

    # Bonus for blocking opponent connections
    opp = Colour.opposite(colour)
    for nx, ny in get_neighbors(x, y, size):
        if board.tiles[nx][ny].colour == opp:
            score += 1.5

    # Bonus for center positions (strategic value)
    center = size // 2
    dist_to_center = abs(x - center) + abs(y - center)
    score += max(0, (size - dist_to_center) / size)

    # Bonus for positions along main diagonal (often strategically important)
    if colour == Colour.RED:
        # RED benefits from positions along y axis progression
        score += 0.5 if x == y else 0
    else:
        # BLUE benefits from positions along x axis progression
        score += 0.5 if x == y else 0

    return score


def select_weighted_move(
    empty_tiles: list[tuple[int, int]],
    board: Board,
    colour: Colour,
) -> tuple[int, int]:
    """
    Select a move with probability weighted by heuristic scores.
    Uses a simplified selection for speed.
    """
    if not empty_tiles:
        return None

    if len(empty_tiles) == 1:
        return empty_tiles[0]

    # For speed, only score a subset of moves
    sample_size = min(8, len(empty_tiles))

    # Get priorities for sampled moves
    import random
    if len(empty_tiles) <= sample_size:
        candidates = empty_tiles
    else:
        candidates = random.sample(empty_tiles, sample_size)

    best_move = candidates[0]
    best_score = -float('inf')

    for move in candidates:
        score = get_move_priority(board, move, colour)
        # Add small random noise to break ties
        score += random.random() * 0.1
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


# Strong opening moves for swap rule evaluation
# These are first moves that are generally considered strong
STRONG_OPENINGS = {
    # Center and near-center positions
    (5, 5), (4, 5), (5, 4), (4, 6), (6, 4), (5, 6), (6, 5),
    (4, 4), (6, 6), (3, 5), (5, 3), (3, 6), (6, 3),
}

# Weak opening moves (edges and corners)
WEAK_OPENINGS = {
    (0, 0), (0, 10), (10, 0), (10, 10),
    (0, 1), (1, 0), (0, 9), (9, 0), (1, 10), (10, 1), (9, 10), (10, 9),
}


def should_swap(opp_move_x: int, opp_move_y: int, board_size: int = 11) -> bool:
    """
    Determine if we should swap based on opponent's opening move.
    Swap if the opening move is strong (we want that position).
    """
    move = (opp_move_x, opp_move_y)

    # Always swap strong openings
    if move in STRONG_OPENINGS:
        return True

    # Never swap weak openings
    if move in WEAK_OPENINGS:
        return False

    # For other moves, use distance from center heuristic
    center = board_size // 2
    dist = abs(opp_move_x - center) + abs(opp_move_y - center)

    # Swap if within 2 tiles of center
    return dist <= 2


# Zobrist hashing for transposition table
_random.seed(42)  # Deterministic for reproducibility

ZOBRIST_TABLE: dict[tuple[int, int, Colour], int] = {}

def _init_zobrist(size: int = 11):
    """Initialize Zobrist hash table."""
    global ZOBRIST_TABLE
    if ZOBRIST_TABLE:
        return
    for i in range(size):
        for j in range(size):
            for colour in [Colour.RED, Colour.BLUE]:
                ZOBRIST_TABLE[(i, j, colour)] = _random.getrandbits(64)

_init_zobrist()


def compute_board_hash(board: Board) -> int:
    """Compute Zobrist hash of board state."""
    h = 0
    for i in range(board.size):
        for j in range(board.size):
            colour = board.tiles[i][j].colour
            if colour is not None:
                h ^= ZOBRIST_TABLE.get((i, j, colour), 0)
    return h


def update_hash(current_hash: int, x: int, y: int, colour: Colour) -> int:
    """Incrementally update hash after a move."""
    return current_hash ^ ZOBRIST_TABLE.get((x, y, colour), 0)