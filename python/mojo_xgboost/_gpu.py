"""ctypes bridge to the optional GPU kernels.

Kept in a separate shared library from the CPU kernels on purpose: the GPU build
links `libmax.so`, and a machine without MAX (or without a card) must still
import this package and get the CPU path. Nothing here raises on absence —
`available()` answers the question and every entry point returns None when the
device is unusable, so callers fall through to `_lib`.

The device is re-checked, never assumed: another process holding the card makes
`DeviceContext()` fail with OOM at call time, not at import time.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

from ._lib import ROOT, addr

LIB = os.path.join(ROOT, "dist", "libmojo-xgboost-gpu.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mxgb_gpu_available": ([], I),
    "mxgb_gpu_quantize": ([I] * 7, I),
    "mxgb_gpu_hist": ([I] * 10, I),
    "mxgb_gpu_predict": ([I] * 6 + [I, I, I, I, F], I),
}

# Below these sizes the PCIe round trip costs more than the kernel saves; the
# thresholds were measured on a 5090 with bench/bench_gpu.py.
MIN_QUANTIZE_CELLS = 2_000_000
MIN_HIST_CELLS = 2_000_000
MIN_PREDICT_WORK = 4_000_000  # rows * trees

_handle: ctypes.CDLL | None = None
_available: bool | None = None


def lib() -> ctypes.CDLL | None:
    global _handle
    if _handle is None:
        if not os.path.exists(LIB):
            return None
        try:
            _handle = ctypes.CDLL(LIB)
        except OSError:
            return None
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_handle, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _handle


def available() -> bool:
    """Whether a usable accelerator exists right now (checked once, then cached)."""
    global _available
    if _available is None:
        handle = lib()
        if handle is None:
            _available = False
        else:
            if os.environ.get("MOJO_XGBOOST_DISABLE_GPU"):
                _available = False
            else:
                _available = bool(handle.mxgb_gpu_available())
    return _available


def quantize(x: np.ndarray, cuts: np.ndarray, max_bin: int):
    """-> (bins [n,d], bins_t [d,n]) or None if the GPU path did not run."""
    n, d = x.shape
    if not available() or n * d < MIN_QUANTIZE_CELLS:
        return None
    handle = lib()
    x = np.ascontiguousarray(x, dtype=np.float64)
    cuts = np.ascontiguousarray(cuts, dtype=np.float64)
    bins = np.empty((n, d), dtype=np.int64)
    bins_t = np.empty((d, n), dtype=np.int64)
    rc = handle.mxgb_gpu_quantize(
        addr(x), addr(cuts), addr(bins), addr(bins_t), n, d, max_bin
    )
    return (bins, bins_t) if rc == 0 else None


def histogram(bins_t: np.ndarray, grad: np.ndarray, hess: np.ndarray,
              row_nodes: np.ndarray | None, max_bin: int, node: int = -1):
    """-> (hist_grad [d,max_bin], hist_hess) or None."""
    d, n = bins_t.shape
    if not available() or n * d < MIN_HIST_CELLS or max_bin > 256:
        return None
    handle = lib()
    bins_t = np.ascontiguousarray(bins_t, dtype=np.int64)
    grad = np.ascontiguousarray(grad, dtype=np.float64)
    hess = np.ascontiguousarray(hess, dtype=np.float64)
    nodes_addr = 0
    if row_nodes is not None:
        row_nodes = np.ascontiguousarray(row_nodes, dtype=np.int64)
        nodes_addr = addr(row_nodes)
    hg = np.zeros((d, max_bin), dtype=np.float64)
    hh = np.zeros((d, max_bin), dtype=np.float64)
    rc = handle.mxgb_gpu_hist(
        addr(bins_t), addr(grad), addr(hess), nodes_addr,
        addr(hg), addr(hh), n, d, max_bin, node,
    )
    return (hg, hh) if rc == 0 else None


def predict(x: np.ndarray, features: np.ndarray, thresholds: np.ndarray,
            defaults: np.ndarray, leaves: np.ndarray, n_trees: int,
            max_nodes: int, base_margin: float = 0.0):
    """-> pred [n] or None."""
    n, d = x.shape
    if not available() or n * n_trees < MIN_PREDICT_WORK:
        return None
    handle = lib()
    x = np.ascontiguousarray(x, dtype=np.float64)
    features = np.ascontiguousarray(features, dtype=np.int64)
    thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
    defaults = np.ascontiguousarray(defaults, dtype=np.int64)
    leaves = np.ascontiguousarray(leaves, dtype=np.float64)
    pred = np.empty(n, dtype=np.float64)
    rc = handle.mxgb_gpu_predict(
        addr(x), addr(features), addr(thresholds), addr(defaults), addr(leaves),
        addr(pred), n, d, n_trees, max_nodes, ctypes.c_double(base_margin),
    )
    return pred if rc == 0 else None
