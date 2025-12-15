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
                "-pthread",
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
        return ctypes.cdll.LoadLibrary(LIB_PATH)


engine = _load_engine()


# ============================================================
# Helper for optional symbol binding
# ============================================================

def _bind_optional(name, argtypes, restype):
    """
    Bind a symbol if it exists in the shared library.
    Returns the bound function or None.
    """
    try:
        fn = getattr(engine, name)
    except AttributeError:
        return None
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


# ============================================================
# Declare required C signatures
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
# Optional C signatures (GRAVE + batching)
# ============================================================

_set_grave_enabled = _bind_optional("set_grave_enabled", [c_int], None)
_set_grave_ref = _bind_optional("set_grave_ref", [c_double], None)
_set_c_puct = _bind_optional("set_c_puct", [c_double], None)

_request_leaves = _bind_optional(
    "request_leaves",
    [c_int, POINTER(c_int), POINTER(c_int), POINTER(c_int)],
    None,
)

_apply_evals_batch = _bind_optional(
    "apply_evals_batch",
    [c_int, POINTER(c_double), POINTER(c_double)],
    None,
)


# ============================================================
# Python wrapper
# ============================================================

class CppMCTS:
    """
    Thin wrapper around the C++ engine with (optional) GRAVE and batching support.
    """

    def __init__(
        self,
        board_size=11,
        sims=300,
        c_puct=1.2,
        use_grave=True,
        grave_ref=0.5,
        batch_size=32,
    ):
        self.N = int(board_size)
        self.sims = int(sims)
        self.c_puct = float(c_puct)
        self.use_grave = bool(use_grave)
        self.grave_ref = float(grave_ref)
        self.batch_size = int(batch_size)

        # Set optional GRAVE parameters in C++ engine (only if symbols exist)
        if _set_c_puct is not None:
            _set_c_puct(c_double(self.c_puct))
        if _set_grave_enabled is not None:
            _set_grave_enabled(c_int(1 if self.use_grave else 0))
        if _set_grave_ref is not None:
            _set_grave_ref(c_double(self.grave_ref))

        # persistent buffers for single-leaf calls
        self._leaf_board = np.zeros(self.N * self.N, dtype=np.int32)
        self._leaf_player = np.zeros(1, dtype=np.int32)
        self._leaf_term = np.zeros(1, dtype=np.int32)

        # persistent buffers for batched calls
        self._batch_boards = np.zeros((self.batch_size, self.N * self.N), dtype=np.int32)
        self._batch_players = np.zeros(self.batch_size, dtype=np.int32)
        self._batch_terms = np.zeros(self.batch_size, dtype=np.int32)

    def reset(self, board_flat: np.ndarray, player: int):
        assert board_flat.size == self.N * self.N
        arr = np.ascontiguousarray(board_flat, dtype=np.int32)

        engine.reset_tree(
            arr.ctypes.data_as(POINTER(c_int)),
            c_int(self.N),
            c_int(int(player)),
        )

    def request_leaf(self):
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
        pri = np.ascontiguousarray(priors, dtype=np.float64)
        engine.apply_eval(
            pri.ctypes.data_as(POINTER(c_double)),
            c_double(float(value)),
        )

    def best_action(self) -> int:
        return int(engine.best_action())

    def set_grave_enabled(self, enabled: bool):
        self.use_grave = bool(enabled)
        if _set_grave_enabled is not None:
            _set_grave_enabled(c_int(1 if self.use_grave else 0))

    def set_grave_ref(self, ref: float):
        self.grave_ref = float(ref)
        if _set_grave_ref is not None:
            _set_grave_ref(c_double(self.grave_ref))

    def set_c_puct(self, c_puct: float):
        self.c_puct = float(c_puct)
        if _set_c_puct is not None:
            _set_c_puct(c_double(self.c_puct))

    def request_leaves(self, B: int):
        if _request_leaves is None:
            raise RuntimeError("[CppMCTS] request_leaves not available. Rebuild .so with batching functions.")
        B = int(B)
        assert 1 <= B <= self.batch_size

        _request_leaves(
            c_int(B),
            self._batch_boards.ctypes.data_as(POINTER(c_int)),
            self._batch_players.ctypes.data_as(POINTER(c_int)),
            self._batch_terms.ctypes.data_as(POINTER(c_int)),
        )

        return (
            self._batch_boards[:B].copy(),
            self._batch_players[:B].copy(),
            self._batch_terms[:B].copy(),
        )

    def apply_evals_batch(self, priors_batch: np.ndarray, values_batch: np.ndarray):
        if _apply_evals_batch is None:
            raise RuntimeError("[CppMCTS] apply_evals_batch not available. Rebuild .so with batching functions.")

        pri = np.ascontiguousarray(priors_batch, dtype=np.float64)
        val = np.ascontiguousarray(values_batch, dtype=np.float64)
        B = int(val.shape[0])

        _apply_evals_batch(
            c_int(B),
            pri.ctypes.data_as(POINTER(c_double)),
            val.ctypes.data_as(POINTER(c_double)),
        )
