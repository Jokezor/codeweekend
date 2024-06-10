import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL("./libhero_functions.so")

# Define the argument and return types for the C++ functions
lib.build_kd_tree.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
]

lib.find_top_scoring_monsters.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
]

lib.find_closest_monster.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_int),
]


# Python functions to wrap the C++ functions
def build_kd_tree(monsters):
    positions = np.array(
        [(monster["x"], monster["y"]) for monster in monsters], dtype=np.float64
    ).flatten()
    tree = np.zeros_like(positions)
    lib.build_kd_tree(positions, len(monsters), tree)
    return tree


def find_top_scoring_monsters(
    hero_pos, monsters, max_distance_squared, top_n=5, ratio=0.5
):
    positions = np.array(
        [(monster["x"], monster["y"]) for monster in monsters], dtype=np.float64
    ).flatten()
    hp = np.array([monster["hp"] for monster in monsters], dtype=np.float64)
    gold = np.array([monster["gold"] for monster in monsters], dtype=np.float64)
    exp = np.array([monster["exp"] for monster in monsters], dtype=np.float64)
    scores = np.zeros(top_n, dtype=np.float64)
    indices = np.zeros(top_n, dtype=np.int32)

    lib.find_top_scoring_monsters(
        hero_pos,
        positions,
        hp,
        gold,
        exp,
        len(monsters),
        max_distance_squared,
        top_n,
        ratio,
        scores,
        indices,
    )
    return [(scores[i], indices[i]) for i in range(top_n) if indices[i] >= 0]


def find_closest_monster(hero_pos, monsters, range_squared):
    positions = np.array(
        [(monster["x"], monster["y"]) for monster in monsters], dtype=np.float64
    ).flatten()
    hp = np.array([monster["hp"] for monster in monsters], dtype=np.float64)
    closest_idx = ctypes.c_int()

    lib.find_closest_monster(
        hero_pos, positions, hp, len(monsters), range_squared, ctypes.byref(closest_idx)
    )
    return closest_idx.value


# The remaining Python code...
