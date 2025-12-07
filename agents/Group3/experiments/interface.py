import os
import ctypes
import subprocess
import shutil
from ctypes import c_int, c_double, POINTER
import numpy as np

# ============================================================
# Locate shared library and source
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_NAME = "hex_mcts_engine.so"
LIB_PATH = os.path.join(HERE, LIB_NAME)
SRC_PATH = os.path.join(HERE, "hex_mcts_engine.cpp")


def _build_hex_mcts_engine():
    """
    Build hex_mcts_engine.so in this directory using g++.
    Will try to install g++ via apt-get if it is missing.
    """
    print(f"[CppMCTS] Building {LIB_NAME} from {SRC_PATH}")

    if not os.path.exists(SRC_PATH):
        raise FileNotFoundError(f"[CppMCTS] C++ source not found at {SRC_PATH}")

    # Ensure we have g++
    if shutil.which("g++") is None:
        print("[CppMCTS] g++ not found, attempting apt-get install g++")
        try:
            subprocess.run(
                ["apt-get", "update"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            subprocess.run(
                ["apt-get", "install", "-y", "g++"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except Exception as e:
            raise RuntimeError(f"[CppMCTS] Failed to install g++: {e}")

    # Compile the shared library
    try:
        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-shared",
                "hex_mcts_engine.cpp",
                "-o",
                "hex_mcts_engine.so",
            ],
            cwd=HERE,
            check=True,
        )
    except Exception as e:
        raise RuntimeError(f"[CppMCTS] Failed to compile hex_mcts_engine.so: {e}")

    if not os.path.exists(LIB_PATH):
        raise RuntimeError("[CppMCTS] hex_mcts_engine.so not produced by compiler")

    print("[CppMCTS] Build complete.")


def _load_engine():
    """
    Try to load the shared library.
    If it fails (missing or invalid ELF), attempt to rebuild and load again.
    """
    try:
        return ctypes.cdll.LoadLibrary(LIB_PATH)
    except OSError as e:
        print(f"[CppMCTS] Initial load failed for {LIB_PATH}: {e}")
        print("[CppMCTS] Attempting in-container rebuild of hex_mcts_engine.so")
        _build_hex_mcts_engine()
        # Second attempt
        return ctypes.cdll.LoadLibrary(LIB_PATH)


engine = _load_engine()

# ============================================================
# Declare C signatures
# ============================================================

# void reset_tree(int* board, int N, int player);
engine.reset_tree.argtypes = [POINTER(c_int), c_int, c_int]
engine.reset_tree.restype = None

# void request_leaf(int* out_board, int* out_player, int* out_is_terminal);
engine.request_leaf.argtypes = [
    POINTER(c_int),
    POINTER(c_int),
    POINTER(c_int),
]
engine.request_leaf.restype = None

# void apply_eval(const double* priors, double value);
engine.apply_eval.argtypes = [POINTER(c_double), c_double]
engine.apply_eval.restype = None

# int best_action();
engine.best_action.argtypes = []
engine.best_action.restype = c_int


# ============================================================
# Python wrapper
# ============================================================

class CppMCTS:
    """
    Thin wrapper around the C++ engine.

    Protocol per move:
      1) reset(board_flat, player)
      2) For each NN evaluation:
           leaf_board, leaf_player, is_term = request_leaf()
           -> evaluate with NN
           apply_eval(priors, value)
      3) action = best_action()
    """

    def __init__(self, board_size=11, sims=300, c_puct=1.2):
        self.N = board_size
        self.sims = sims
        self.c_puct = c_puct

        # persistent buffers for C calls
        self._leaf_board = np.zeros(self.N * self.N, dtype=np.int32)
        self._leaf_player = np.zeros(1, dtype=np.int32)
        self._leaf_term = np.zeros(1, dtype=np.int32)

    def reset(self, board_flat: np.ndarray, player: int):
        """
        Initialise C++ tree at root for given board and player.
        board_flat is a flat N*N int32 array (0 empty, 1 red, 2 blue).
        player is 1 or 2.
        """
        assert board_flat.size == self.N * self.N
        arr = np.ascontiguousarray(board_flat, dtype=np.int32)

        engine.reset_tree(
            arr.ctypes.data_as(POINTER(c_int)),
            c_int(self.N),
            c_int(player),
        )

    def request_leaf(self):
        """
        Ask C++ engine for a leaf position to evaluate.

        Returns:
            flat_board (np.ndarray int32 shape (N*N,))
            player_to_move (int, 1 or 2)
            is_terminal (int, 0 or 1)
        """
        engine.request_leaf(
            self._leaf_board.ctypes.data_as(POINTER(c_int)),
            self._leaf_player.ctypes.data_as(POINTER(c_int)),
            self._leaf_term.ctypes.data_as(POINTER(c_int)),
        )

        return (
            self._leaf_board.copy(),
            int(self._leaf_player[0]),
            int(self._leaf_term[0]),
        )

    def apply_eval(self, priors: np.ndarray, value: float):
        """
        Feed NN priors and scalar value back into C++ at the last leaf.

        priors: np.ndarray float64, shape (N*N,)
        value: float
        """
        pri = np.ascontiguousarray(priors, dtype=np.float64)
        engine.apply_eval(
            pri.ctypes.data_as(POINTER(c_double)),
            c_double(value),
        )

    def best_action(self) -> int:
        """
        Return the chosen action index at the root (0..N*N - 1).
        """
        return int(engine.best_action())