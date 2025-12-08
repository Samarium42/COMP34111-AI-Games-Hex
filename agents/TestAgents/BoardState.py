# agents/Group25/BoardState.py

class BoardState:
    EMPTY = 0
    RED = 1
    BLUE = 2

    # Neighbour coordinates in a Hex grid
    NEIGHBOURS = [(1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)]

    def __init__(self, size=11):
        self.size = size
        self.grid = [[0]*size for _ in range(size)]

        # Maintain a list of legal empty cells
        self.empty_cells = [(x,y) for x in range(size) for y in range(size)]

        # Union-Find structure for fast connectivity checks
        n = size * size
        self.parent_red = list(range(n + 2))   # +2 for top and bottom virtual nodes
        self.parent_blue = list(range(n + 2))  # +2 for left and right virtual nodes

        self.TOP = n
        self.BOTTOM = n+1
        self.LEFT = n
        self.RIGHT = n+1

        self.winner = None

    def clone(self):
        """Fast board cloning."""
        new = BoardState(self.size)
        new.grid = [row[:] for row in self.grid]
        new.empty_cells = self.empty_cells[:]   # shallow copy is fine
        new.parent_red = self.parent_red[:]
        new.parent_blue = self.parent_blue[:]
        new.winner = self.winner
        return new

    # --------------------
    # UNION-FIND FUNCTIONS
    # --------------------
    def find(self, parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, parent, a, b):
        ra = self.find(parent, a)
        rb = self.find(parent, b)
        if ra != rb:
            parent[rb] = ra

    # --------------------
    # PLAY MOVE
    # --------------------
    def play(self, x, y, colour):
        """Apply a move (colour is BoardState.RED or BLUE)."""

        if self.grid[x][y] != 0:
            raise ValueError("Invalid move - cell already occupied")

        self.grid[x][y] = colour
        self.empty_cells.remove((x, y))

        # Convert (x, y) to index in DSU
        index = x * self.size + y

        if colour == BoardState.RED:
            parent = self.parent_red

            # Connect to top or bottom virtual nodes
            if x == 0:
                self.union(parent, index, self.TOP)
            if x == self.size - 1:
                self.union(parent, index, self.BOTTOM)

            # Connect to same-colour neighbours
            for dx, dy in BoardState.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == BoardState.RED:
                        n_index = nx * self.size + ny
                        self.union(parent, index, n_index)

            # Check for win
            if self.find(parent, self.TOP) == self.find(parent, self.BOTTOM):
                self.winner = BoardState.RED

        else:  # BLUE
            parent = self.parent_blue

            if y == 0:
                self.union(parent, index, self.LEFT)
            if y == self.size - 1:
                self.union(parent, index, self.RIGHT)

            for dx, dy in BoardState.NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.grid[nx][ny] == BoardState.BLUE:
                        n_index = nx * self.size + ny
                        self.union(parent, index, n_index)

            if self.find(parent, self.LEFT) == self.find(parent, self.RIGHT):
                self.winner = BoardState.BLUE

    # --------------------
    # UTILITY
    # --------------------
    def get_legal_moves(self):
        return self.empty_cells

    def is_terminal(self):
        return self.winner is not None

    def get_winner(self):
        return self.winner
    
    def get_empty_cells(self):
        return [(x,y) for x in range(self.size) for y in range(self.size) if self.tiles[x][y] == 0]
