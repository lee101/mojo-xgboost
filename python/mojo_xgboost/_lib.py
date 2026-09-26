"""ctypes bridge to the Mojo histogram kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-xgboost.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mxgb_quantize": ([I, I, I, I, I, I], None),
    "mxgb_build_tree": ([I] * 14 + [I, I, I, I] + [F] * 5 + [I], I),
    "mxgb_predict": ([I, I, I, I, I, I, I, I, I, I, F, I], None),
    "mxgb_predict_range": ([I, I, I, I, I, I, I, I, I, I, F, I, I], None),
    "mxgb_predict_add": ([I, I, I, I, I, I, I, I, I], None),
    "mxgb_predict_add_range": ([I, I, I, I, I, I, I, I, I, I], None),
    "mxgb_predict_leaf": ([I, I, I, I, I, I, I, I, I, I], None),
    "mxgb_predict_leaf_range": ([I] * 11, None),
}

# Per-row tree walks are latency-bound on dependent gathers, not bandwidth-bound,
# so they scale with cores. Below this many row-trees the thread hand-off costs
# more than the walk.
CHUNK_WORK = 200_000

_handle: ctypes.CDLL | None = None


def build() -> str:
    sources = [os.path.join(ROOT, "src", "capi.mojo")]
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < max(
        os.path.getmtime(path) for path in sources
    )
    if stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode or not os.path.exists(LIB):
            raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _handle
    if _handle is None:
        _handle = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_handle, name)
            function.argtypes = argtypes
            function.restype = restype
    return _handle


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if array.size and array.ctypes.data == 0:
        raise ValueError("native buffers must have a non-null address")
    return array.ctypes.data


def f64(data, *, copy: bool = False) -> np.ndarray:
    array = np.asarray(data)
    if array.dtype.kind not in "biuf":
        raise TypeError("values must be real numeric data")
    if array.dtype.kind in "iu" and array.size:
        limit = 1 << 53
        if np.any(array > limit) or np.any(array < -limit):
            raise ValueError("integer values outside [-2**53, 2**53] lose float64 precision")
    if copy:
        return np.array(array, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(data, *, copy: bool = False) -> np.ndarray:
    array = np.asarray(data)
    if array.dtype.kind not in "biu":
        raise TypeError("indices must be integer data")
    if array.dtype.kind == "u" and array.size and np.any(array > np.iinfo(np.int64).max):
        raise ValueError("unsigned integer value does not fit in int64")
    if copy:
        return np.array(array, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(array, dtype=np.int64)


def run_row_kernel(name: str, arguments, rows: int, n_threads: int, work: int):
    """Call a per-row kernel over `rows`, fanning out over `n_threads` chunks.

    Every row is independent and each chunk writes a disjoint output range, so
    the result does not depend on how the rows are split.
    """
    function = getattr(lib(), name)
    workers = min(n_threads, rows, os.cpu_count() or 1) if n_threads > 0 else 1
    if workers <= 1 or work < CHUNK_WORK:
        function(*arguments, 0, rows)
        return
    step = -(-rows // workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(
            pool.map(
                lambda part: function(
                    *arguments, part * step, min((part + 1) * step, rows)
                ),
                range(workers),
            )
        )
