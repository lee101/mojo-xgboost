"""GPU kernels against the CPU kernels they replace.

Skipped wholesale when no accelerator is usable, which is the normal case on CI.
The point of each test is that the GPU path is not a *different* implementation:
quantize must be bit-identical (it is integer output from the same comparisons),
the histogram may differ only by float reassociation, and prediction must match
the CPU traversal exactly because it is the same integer path through the trees.
"""
from __future__ import annotations

import numpy as np
import pytest

from mojo_xgboost import _gpu, _lib

pytestmark = pytest.mark.skipif(not _gpu.available(), reason="no usable GPU")


def synth(n=60_000, d=48, max_bin=64, seed=0, nan_frac=0.01):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.normal(size=(n, d)))
    if nan_frac:
        x[rng.random((n, d)) < nan_frac] = np.nan
    cuts = np.ascontiguousarray(np.sort(rng.normal(size=(d, max_bin - 1)), axis=1))
    return x, cuts, max_bin


def cpu_quantize(x, cuts, max_bin):
    n, d = x.shape
    bins = np.empty((n, d), dtype=np.int64)
    _lib.lib().mxgb_quantize(_lib.addr(x), _lib.addr(cuts), _lib.addr(bins), n, d, max_bin)
    return bins


def force(fn, *args, **kwargs):
    """Call a GPU entry point ignoring the size threshold."""
    saved = (_gpu.MIN_QUANTIZE_CELLS, _gpu.MIN_HIST_CELLS, _gpu.MIN_PREDICT_WORK)
    _gpu.MIN_QUANTIZE_CELLS = _gpu.MIN_HIST_CELLS = _gpu.MIN_PREDICT_WORK = 0
    try:
        return fn(*args, **kwargs)
    finally:
        _gpu.MIN_QUANTIZE_CELLS, _gpu.MIN_HIST_CELLS, _gpu.MIN_PREDICT_WORK = saved


def test_quantize_matches_cpu_bit_for_bit():
    x, cuts, max_bin = synth()
    got = force(_gpu.quantize, x, cuts, max_bin)
    assert got is not None
    bins, bins_t = got
    assert np.array_equal(bins, cpu_quantize(x, cuts, max_bin))
    assert np.array_equal(bins_t, bins.T)  # feature-major view for coalesced reads


def test_histogram_matches_bincount():
    x, cuts, max_bin = synth()
    bins = cpu_quantize(x, cuts, max_bin)
    bins_t = np.ascontiguousarray(bins.T)
    rng = np.random.default_rng(1)
    n = x.shape[0]
    grad = np.ascontiguousarray(rng.normal(size=n))
    hess = np.ascontiguousarray(rng.random(n) + 0.5)
    got = force(_gpu.histogram, bins_t, grad, hess, None, max_bin)
    assert got is not None
    hg, hh = got
    for f in range(x.shape[1]):
        ok = bins[:, f] >= 0
        ref_g = np.bincount(bins[ok, f], weights=grad[ok], minlength=max_bin)[:max_bin]
        ref_h = np.bincount(bins[ok, f], weights=hess[ok], minlength=max_bin)[:max_bin]
        assert np.allclose(hg[f], ref_g, rtol=1e-9, atol=1e-9)
        assert np.allclose(hh[f], ref_h, rtol=1e-9, atol=1e-9)


def test_histogram_respects_the_node_filter():
    x, cuts, max_bin = synth(n=20_000, d=8)
    bins = cpu_quantize(x, cuts, max_bin)
    bins_t = np.ascontiguousarray(bins.T)
    rng = np.random.default_rng(2)
    n = x.shape[0]
    grad = np.ascontiguousarray(rng.normal(size=n))
    hess = np.ascontiguousarray(np.ones(n))
    row_nodes = np.ascontiguousarray(rng.integers(0, 4, size=n))
    got = force(_gpu.histogram, bins_t, grad, hess, row_nodes, max_bin, 2)
    assert got is not None
    hg, _ = got
    sel = row_nodes == 2
    for f in range(x.shape[1]):
        ok = sel & (bins[:, f] >= 0)
        ref = np.bincount(bins[ok, f], weights=grad[ok], minlength=max_bin)[:max_bin]
        assert np.allclose(hg[f], ref, rtol=1e-9, atol=1e-9)


def test_predict_matches_cpu():
    rng = np.random.default_rng(3)
    n, d, n_trees, depth = 20_000, 16, 32, 5
    max_nodes = (1 << (depth + 1)) - 1
    x = np.ascontiguousarray(rng.normal(size=(n, d)))
    x[rng.random((n, d)) < 0.02] = np.nan
    features = np.ascontiguousarray(rng.integers(0, d, size=(n_trees, max_nodes)))
    features[:, max_nodes // 2 :] = -1
    thresholds = np.ascontiguousarray(rng.normal(size=(n_trees, max_nodes)))
    defaults = np.ascontiguousarray(rng.integers(0, 2, size=(n_trees, max_nodes)))
    leaves = np.ascontiguousarray(rng.normal(size=(n_trees, max_nodes)) * 0.01)
    cpu = np.empty(n)
    _lib.lib().mxgb_predict(
        _lib.addr(x), _lib.addr(features), _lib.addr(thresholds), _lib.addr(defaults),
        _lib.addr(leaves), _lib.addr(cpu), n, d, n_trees, max_nodes, 0.25, 0,
    )
    gpu = force(_gpu.predict, x, features, thresholds, defaults, leaves, n_trees, max_nodes, 0.25)
    assert gpu is not None
    assert np.allclose(gpu, cpu, rtol=1e-12, atol=1e-12)


def test_below_threshold_declines_rather_than_transferring():
    """Small inputs must return None so the caller stays on the CPU path."""
    x, cuts, max_bin = synth(n=100, d=4)
    assert _gpu.quantize(x, cuts, max_bin) is None


def test_max_bin_over_shared_capacity_declines():
    x, cuts, max_bin = synth(n=1000, d=4, max_bin=512)
    bins = cpu_quantize(x, cuts, max_bin)
    bins_t = np.ascontiguousarray(bins.T)
    grad = np.ascontiguousarray(np.ones(1000))
    assert force(_gpu.histogram, bins_t, grad, grad, None, 512) is None
