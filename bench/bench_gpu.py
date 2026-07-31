#!/usr/bin/env python3
"""GPU vs CPU for the three wide stages. Markdown table, sized for a PR body.

Timing includes the host<->device copies, because that is what a caller pays.
Anything that does not beat the CPU kernel including transfer does not belong on
the GPU, and the thresholds in `_gpu.py` are set from this table.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mojo_xgboost import _gpu, _lib  # noqa: E402


def timeit(fn, reps=3):
    """Best-of-N wall clock, with the result checked.

    A GPU entry point returns None when the device refuses the allocation, and
    timing that failure produces a spectacular fake speedup — so the result is
    asserted, not discarded.
    """
    first = fn()
    if first is None:
        return None
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        got = fn()
        if got is None:
            return None
        best = min(best, time.perf_counter() - t)
    return best


def fmt(seconds):
    return "n/a" if seconds is None else f"{seconds * 1e3:.1f} ms"


def ratio(cpu, gpu):
    return "n/a" if (cpu is None or gpu is None) else f"{cpu / gpu:.2f}x"


def cpu_quantize(x, cuts, max_bin):
    n, d = x.shape
    bins = np.empty((n, d), dtype=np.int64)
    _lib.lib().mxgb_quantize(_lib.addr(x), _lib.addr(cuts), _lib.addr(bins), n, d, max_bin)
    return bins


def cpu_hist(bins, grad, hess, max_bin):
    """Fast CPU histogram: bincount per feature, not `np.add.at`.

    `np.add.at` is ~50x slower than bincount and using it as the baseline would
    flatter the GPU by an order of magnitude.
    """
    d = bins.shape[1]
    hg = np.empty((d, max_bin))
    hh = np.empty((d, max_bin))
    for f in range(d):
        col = bins[:, f]
        hg[f] = np.bincount(col, weights=grad, minlength=max_bin)[:max_bin]
        hh[f] = np.bincount(col, weights=hess, minlength=max_bin)[:max_bin]
    return hg, hh


def main():
    if not _gpu.available():
        print("no GPU available")
        return
    rng = np.random.default_rng(0)
    max_bin = 64
    print("| stage | shape | CPU | GPU (Mojo) | speedup |")
    print("|---|---|---|---|---|")
    print("<!-- CPU = Mojo SIMD kernel for quantize/predict, NumPy bincount for histogram -->")
    for n, d in ((100_000, 32), (500_000, 64), (2_000_000, 64)):
        x = np.ascontiguousarray(rng.normal(size=(n, d)))
        cuts = np.ascontiguousarray(np.sort(rng.normal(size=(d, max_bin - 1)), axis=1))
        c = timeit(lambda: cpu_quantize(x, cuts, max_bin))
        g = timeit(lambda: _gpu.quantize(x, cuts, max_bin))
        print(f"| quantize | {n}x{d} | {fmt(c)} | {fmt(g)} | {ratio(c, g)} |")

        bins = cpu_quantize(x, cuts, max_bin)
        bins_t = np.ascontiguousarray(bins.T)
        grad = np.ascontiguousarray(rng.normal(size=n))
        hess = np.ascontiguousarray(rng.random(n) + 0.5)

        c = timeit(lambda: cpu_hist(bins, grad, hess, max_bin))
        g = timeit(lambda: _gpu.histogram(bins_t, grad, hess, None, max_bin))
        gh = _gpu.histogram(bins_t, grad, hess, None, max_bin)
        err = "n/a"
        if gh is not None:
            ref = cpu_hist(bins, grad, hess, max_bin)
            err = f"{np.abs(gh[0] - ref[0]).max():.1e}"
        print(f"| histogram | {n}x{d} | {fmt(c)} | {fmt(g)} | {ratio(c, g)} | max err {err}")

    # prediction: rows x trees is the work term
    n, d, n_trees, depth = 500_000, 32, 200, 6
    max_nodes = (1 << (depth + 1)) - 1
    x = np.ascontiguousarray(rng.normal(size=(n, d)))
    features = np.ascontiguousarray(rng.integers(0, d, size=(n_trees, max_nodes)))
    features[:, (max_nodes // 2):] = -1
    thresholds = np.ascontiguousarray(rng.normal(size=(n_trees, max_nodes)))
    defaults = np.ascontiguousarray(np.zeros((n_trees, max_nodes), dtype=np.int64))
    leaves = np.ascontiguousarray(rng.normal(size=(n_trees, max_nodes)) * 0.01)
    pred = np.empty(n)
    def run_cpu_predict():
        _lib.lib().mxgb_predict(
            _lib.addr(x), _lib.addr(features), _lib.addr(thresholds), _lib.addr(defaults),
            _lib.addr(leaves), _lib.addr(pred), n, d, n_trees, max_nodes, 0.0, 0)
        return pred

    cpu = timeit(run_cpu_predict)
    gpu = timeit(lambda: _gpu.predict(x, features, thresholds, defaults, leaves, n_trees, max_nodes))
    print(f"| predict | {n} rows x {n_trees} trees | {fmt(cpu)} | {fmt(gpu)} | {ratio(cpu, gpu)} |")


if __name__ == "__main__":
    main()
