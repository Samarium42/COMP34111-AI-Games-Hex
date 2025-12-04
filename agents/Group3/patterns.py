"""
Minimal, original pattern recogniser for Hex.

The goal is not to replicate expert template systems.
Instead, we include three simple local structures:

1. Bridge carriers: two stones separated by (1,1) have two key empty cells.
2. Edge pressure: a stone near the relevant edge benefits from one-step extensions.
3. Forcing moves: an empty cell adjacent to two friendly stones is a local threat.

All definitions are geometric only and do not borrow from external templates.
"""

from typing import List, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from src.Board import Board
from src.Colour import Colour


class PatternType(Enum):
    BRIDGE = 1
    EDGE_PRESSURE = 2
    FORCING = 3


@dataclass
class Pattern:
    type: PatternType
    key_cells: Set[Tuple[int, int]]
    priority: int
    colour: Colour


class PatternMatcher:
    NEIGH = [(-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0)]

    def neighbors(self, x, y, N):
        for dx, dy in self.NEIGH:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                yield nx, ny

    def find(self, board: Board, colour: Colour) -> List[Pattern]:
        out = []
        out.extend(self._bridges(board, colour))
        out.extend(self._edge_pressure(board, colour))
        out.extend(self._forcing(board, colour))
        return out

    # 1. Bridges
    def _bridges(self, board: Board, colour: Colour) -> List[Pattern]:
        N = board.size
        stones = []

        for i in range(N):
            for j in range(N):
                if board.tiles[i][j].colour == colour:
                    stones.append((i, j))

        out = []
        for (x1, y1) in stones:
            for (x2, y2) in stones:
                if (x2, y2) <= (x1, y1):
                    continue

                if abs(x2 - x1) == 1 and abs(y2 - y1) == 1:
                    # Candidate bridge
                    carriers = set()
                    for nx, ny in self.neighbors(x1, y1, N):
                        if board.tiles[nx][ny].colour is None:
                            if (nx, ny) in self.neighbors(x2, y2, N):
                                carriers.add((nx, ny))
                    if len(carriers) == 2:
                        out.append(Pattern(
                            type=PatternType.BRIDGE,
                            key_cells=carriers,
                            priority=1,
                            colour=colour
                        ))
        return out

    # 2. Edge pressure
    def _edge_pressure(self, board: Board, colour: Colour) -> List[Pattern]:
        N = board.size
        out = []

        if colour == Colour.RED:
            # Red presses top and bottom
            edges = [0, N - 1]
            for row in edges:
                for col in range(N):
                    if board.tiles[row][col].colour == colour:
                        cands = set()
                        for nx, ny in self.neighbors(row, col, N):
                            if board.tiles[nx][ny].colour is None:
                                cands.add((nx, ny))
                        if cands:
                            out.append(Pattern(
                                type=PatternType.EDGE_PRESSURE,
                                key_cells=cands,
                                priority=2,
                                colour=colour
                            ))

        else:
            # Blue presses left and right
            edges = [0, N - 1]
            for r in range(N):
                for c in edges:
                    if board.tiles[r][c].colour == colour:
                        cands = set()
                        for nx, ny in self.neighbors(r, c, N):
                            if board.tiles[nx][ny].colour is None:
                                cands.add((nx, ny))
                        if cands:
                            out.append(Pattern(
                                type=PatternType.EDGE_PRESSURE,
                                key_cells=cands,
                                priority=2,
                                colour=colour
                            ))

        return out

    # 3. Forcing moves
    def _forcing(self, board: Board, colour: Colour) -> List[Pattern]:
        N = board.size
        out = []

        for i in range(N):
            for j in range(N):
                if board.tiles[i][j].colour is None:
                    adj = sum(
                        1 for nx, ny in self.neighbors(i, j, N)
                        if board.tiles[nx][ny].colour == colour
                    )
                    if adj >= 2:
                        out.append(Pattern(
                            type=PatternType.FORCING,
                            key_cells={(i, j)},
                            priority=1,
                            colour=colour
                        ))
        return out
