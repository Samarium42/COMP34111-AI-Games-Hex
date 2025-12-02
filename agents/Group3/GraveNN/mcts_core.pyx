# =============================
#  mcts_core.pyx  —  Cython MCTS
# =============================

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# These are Python classes → must be typed as "object"
from src.Board import Board
from src.Move import Move
from src.Colour import Colour


# ===============================================
# 1. Hex GameState (Python objects + C fast loops)
# ===============================================
cdef class HexState:
    cdef public object board       # Board*
    cdef public object player      # Colour*
    cdef public int board_size

    def __init__(self, board, player):
        self.board = board
        self.player = player
        self.board_size = board.size

    cpdef HexState clone(self):
        return HexState(self.board.copy(), self.player)

    cpdef list legal_moves(self):
        cdef int N = self.board_size
        cdef list moves = []
        cdef int x,y

        for x in range(N):
            for y in range(N):
                if self.board.tiles[x][y].colour is None:
                    moves.append(x*N + y)

        if self.turn_number() == 2:
            moves.append(self.swap_index())

        return moves

    cpdef HexState play(self, int action_index):
        cdef HexState new_state = self.clone()
        cdef int x,y

        if action_index == self.swap_index():
            new_state.board.swap_colours()
        else:
            x = action_index // self.board_size
            y = action_index % self.board_size
            new_state.board.set_tile_colour(x, y, self.player)

        new_state.player = Colour.RED if self.player == Colour.BLUE else Colour.BLUE
        return new_state

    cpdef int swap_index(self):
        return self.board_size * self.board_size

    cpdef int turn_number(self):
        cdef int filled = 0
        cdef int x,y,N=self.board_size

        for x in range(N):
            for y in range(N):
                if self.board.tiles[x][y].colour is not None:
                    filled+=1
        return filled+1

    cpdef bint is_terminal(self):
        return self.board.has_ended()

    cpdef double result(self):
        winner=self.board.get_winner()
        if winner is None: return 0.0
        if winner==Colour.RED: return 1.0
        return -1.0


# =======================
# 2. Search Tree Node
# =======================
cdef class Node:
    cdef public object state
    cdef public Node parent
    cdef public double prior
    cdef dict children
    cdef public int N
    cdef public double W,Q

    def __init__(self,state,parent, float prior):
        self.state=state
        self.parent=parent
        self.prior=prior
        self.children={}
        self.N=0; self.W=0; self.Q=0

    cpdef expand(self, list legal, np.ndarray priors):
        cdef int a
        for a in legal:
            self.children[a]=Node(self.state.play(a),self,float(priors[a]))

    cdef void backup(self,double value):
        cdef Node node=self
        cdef double v=value
        while node is not None:
            node.N+=1
            node.W+=v
            node.Q=node.W/node.N
            v=-v
            node=node.parent

    cpdef select_child(self,double c_puct):
        cdef double best=-1e12
        cdef int best_a=-1
        cdef Node best_child=None

        cdef double total=0
        for c in self.children.values():
            total+=c.N
        total=sqrt(total+1e-8)

        for a,c in self.children.items():
            U = c_puct*c.prior*total/(1+c.N)
            score=c.Q+U
            if score>best:
                best=score; best_a=a; best_child=c
        return best_a,best_child


# =======================
# 3. MCTS Engine
# =======================
cdef class MCTS:
    cdef public int sims
    cdef public double c_puct
    cdef public object net
    cdef public object device

    def __init__(self,net,int sims=200,double c_puct=1.2,device="cpu"):
        self.net=net; self.sims=sims
        self.c_puct=c_puct; self.device=device

    cpdef np.ndarray run(self,HexState root_state):
        cdef Node root=Node(root_state,None,1.0)
        self._expand(root)

        cdef int i
        for i in range(self.sims):
            node=root
            while node.children and not node.state.is_terminal():
                _,node=node.select_child(self.c_puct)

            if node.state.is_terminal():
                node.backup(node.state.result())
                continue

            v=self._expand(node)
            node.backup(v)

        cdef int N=root.state.board_size
        cdef np.ndarray counts = np.zeros(N*N+1,np.float32)
        for a,c in root.children.items():
            counts[a]=c.N
        return counts

    cpdef double _expand(self,Node node):
        if node.state.is_terminal():
            return node.state.result()

        import torch
        with torch.no_grad():
            x=node.state.encode().unsqueeze(0).to(self.device)
            logits,value=self.net(x)
            priors=torch.softmax(logits[0],dim=0).cpu().numpy()
            value=float(value.item())

        legal=node.state.legal_moves()
        mask=np.zeros_like(priors)
        mask[legal]=1
        priors=priors*mask
        priors/=priors.sum() if priors.sum()>0 else mask.sum()

        node.expand(legal,priors)
        return value
