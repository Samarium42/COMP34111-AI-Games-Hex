import os
import ctypes
from ctypes import c_int, c_double, POINTER
import numpy as np


# ============================================================
# Load shared library
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_NAME = "hex_mcts_engine.so"   # make sure your Makefile builds this name
LIB_PATH = os.path.join(HERE, LIB_NAME)

engine = ctypes.cdll.LoadLibrary(LIB_PATH)

# C signatures:
#   void reset_tree(int* board, int N, int player);
#   void request_leaf(int* out_board, int* out_player, int* out_is_terminal);
#   void apply_eval(const double* priors, double value);
#   int  best_action();

engine.reset_tree.argtypes = [POINTER(c_int), c_int, c_int]
engine.reset_tree.restype  = None

engine.request_leaf.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
engine.request_leaf.restype  = None

engine.apply_eval.argtypes = [POINTER(c_double), c_double]
engine.apply_eval.restype  = None

engine.best_action.argtypes = []
engine.best_action.restype  = c_int


# ============================================================
# Python wrapper
# ============================================================

class CppMCTS:
    """
    Thin wrapper around the C++ engine.

    Protocol:
      1) reset(board_flat, player)
      2) repeat:
           leaf_board, leaf_player, is_term = request_leaf()
           -> NN eval in Python
           apply_eval(priors, value)
      3) action = best_action()
    """

    def __init__(self, board_size=11, sims=300, c_puct=1.2):
        self.N = board_size
        self.sims = sims
        self.c_puct = c_puct

        # persistent buffers for C calls
        self._leaf_board  = np.zeros(self.N * self.N, dtype=np.int32)
        self._leaf_player = np.zeros(1, dtype=np.int32)
        self._leaf_term   = np.zeros(1, dtype=np.int32)

    # --------------------------------------------------------
    # Step 0: initialise C++ tree at the root
    # --------------------------------------------------------
    def reset(self, board_flat: np.ndarray, player: int):
        assert board_flat.size == self.N * self.N
        arr = np.ascontiguousarray(board_flat, dtype=np.int32)

        engine.reset_tree(
            arr.ctypes.data_as(POINTER(c_int)),
            c_int(self.N),
            c_int(player),
        )

    # --------------------------------------------------------
    # Step 1: ask C++ for a leaf to evaluate
    # --------------------------------------------------------
    def request_leaf(self):
        engine.request_leaf(
            self._leaf_board.ctypes.data_as(POINTER(c_int)),
            self._leaf_player.ctypes.data_as(POINTER(c_int)),
            self._leaf_term.ctypes.data_as(POINTER(c_int)),
        )

        return (
            self._leaf_board.copy(),       # flat board (N*N,)
            int(self._leaf_player[0]),     # 1 or 2
            int(self._leaf_term[0]),       # 0 or 1
        )

    # --------------------------------------------------------
    # Step 2: feed NN priors + value back into C++
    # --------------------------------------------------------
    def apply_eval(self, priors: np.ndarray, value: float):
        pri = np.ascontiguousarray(priors, dtype=np.float64)
        engine.apply_eval(
            pri.ctypes.data_as(POINTER(c_double)),
            c_double(value),
        )

    # --------------------------------------------------------
    # Step 3: read best action from root
    # --------------------------------------------------------
    def best_action(self) -> int:
        return int(engine.best_action())
