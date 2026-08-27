import numpy as np

import mojo_xgboost as xgb
from mojo_xgboost import _gpu
from mojo_xgboost._lib import addr, lib
from mojo_xgboost.training import _cuts, _quantize


def test_quantize_matches_left_searchsorted():
    x = np.array(
        [[-2.0, 0.5], [0.0, 2.0], [1.0, np.nan], [4.0, -3.0]],
        dtype=np.float64,
    )
    cuts = np.array([[0.0, 1.0, np.inf], [-1.0, 1.0, np.inf]])
    actual = _quantize(x, cuts, 4)
    expected = np.empty_like(actual)
    for row in range(x.shape[0]):
        for feature in range(x.shape[1]):
            expected[row, feature] = (
                -1
                if np.isnan(x[row, feature])
                else np.searchsorted(cuts[feature], x[row, feature], side="left")
            )
    assert np.array_equal(actual, expected)


def test_quantile_cuts_are_sorted_and_padded():
    x = np.array([[0.0, np.nan], [0.0, 3.0], [2.0, 3.0], [4.0, 3.0]])
    cuts = _cuts(x, 8)
    assert cuts.shape == (2, 7)
    for row in cuts:
        finite = row[np.isfinite(row)]
        assert np.all(np.diff(finite) >= 0)
    assert np.isinf(cuts[0]).any()
    assert cuts[1, 0] == 3.0
    assert np.isinf(cuts[1, 1:]).all()


def test_vectorized_quantile_cuts_match_featurewise_reference():
    rng = np.random.default_rng(29)
    x = rng.integers(-3, 4, size=(19, 5)).astype(np.float64)
    x[::4, 1] = np.nan
    x[:, 3] = np.nan
    max_bin = 9
    actual = _cuts(x, max_bin)
    expected = np.full_like(actual, np.inf)
    probabilities = np.arange(1, max_bin, dtype=np.float64) / max_bin
    for feature in range(x.shape[1]):
        values = x[:, feature]
        values = values[~np.isnan(values)]
        if values.size:
            quantiles = np.unique(
                np.quantile(values, probabilities, method="inverted_cdf")
            )
            expected[feature, : quantiles.size] = quantiles
    assert np.array_equal(actual, expected)


def test_quantize_simd_and_scalar_tails_match_searchsorted():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(17, 5))
    x[2, 1] = np.nan
    x[11, 4] = np.nan
    cuts = _cuts(x, 7)
    actual = _quantize(x, cuts, 7)
    expected = np.empty_like(actual)
    for row in range(x.shape[0]):
        for feature in range(x.shape[1]):
            expected[row, feature] = (
                -1
                if np.isnan(x[row, feature])
                else np.searchsorted(cuts[feature], x[row, feature], side="left")
            )
    assert np.array_equal(actual, expected)


def test_one_tree_has_reference_newton_leaves():
    x = np.arange(8, dtype=np.float64).reshape(-1, 1)
    y = np.r_[np.zeros(4), np.ones(4)]
    cuts = _cuts(x, 8)
    bins = _quantize(x, cuts, 8)
    grad = np.full(8, 0.5)
    grad[4:] = -0.5
    hess = np.ones(8)
    nodes = 3
    features = np.empty(nodes, dtype=np.int64)
    split_bins = np.empty(nodes, dtype=np.int64)
    defaults = np.empty(nodes, dtype=np.int64)
    leaves = np.empty(nodes)
    gains = np.empty(nodes)
    covers = np.empty(nodes)
    row_node = np.empty(8, dtype=np.int64)
    node_grad = np.empty(nodes)
    node_hess = np.empty(nodes)
    hist_grad = np.empty(nodes * 8)
    hist_hess = np.empty(nodes * 8)
    split_count = lib().mxgb_build_tree(
        addr(bins), addr(grad), addr(hess), addr(features), addr(split_bins),
        addr(defaults), addr(leaves), addr(gains), addr(covers), addr(row_node),
        addr(node_grad), addr(node_hess), addr(hist_grad), addr(hist_hess),
        8, 1, 8, 1, 1.0, 0.0, 0.0, 0.0, 0.0, 1,
    )
    assert split_count == 1
    assert features[0] == 0
    assert cuts[0, split_bins[0]] == 3.0
    assert leaves[1] == -0.5
    assert leaves[2] == 0.5
    assert covers.tolist() == [8.0, 4.0, 4.0]


def _build_random_tree(
    bins: np.ndarray,
    grad: np.ndarray,
    hess: np.ndarray,
    *,
    max_bin: int,
    max_depth: int,
    n_threads: int,
) -> tuple[np.ndarray, ...]:
    n, d = bins.shape
    nodes = (1 << (max_depth + 1)) - 1
    features = np.empty(nodes, dtype=np.int64)
    split_bins = np.empty(nodes, dtype=np.int64)
    defaults = np.empty(nodes, dtype=np.int64)
    leaves = np.empty(nodes)
    gains = np.empty(nodes)
    covers = np.empty(nodes)
    row_node = np.empty(n, dtype=np.int64)
    node_grad = np.empty(nodes)
    node_hess = np.empty(nodes)
    hist_grad = np.empty(nodes * d * max_bin)
    hist_hess = np.empty(nodes * d * max_bin)
    lib().mxgb_build_tree(
        addr(bins), addr(grad), addr(hess), addr(features), addr(split_bins),
        addr(defaults), addr(leaves), addr(gains), addr(covers), addr(row_node),
        addr(node_grad), addr(node_hess), addr(hist_grad), addr(hist_hess),
        n, d, max_bin, max_depth, 1.0, 1.0, 0.0, 0.0, 0.0, n_threads,
    )
    return (
        features, split_bins, defaults, leaves, gains, covers, row_node,
        node_grad, node_hess, hist_grad, hist_hess,
    )


def test_parallel_tree_threshold_and_histogram_simd_tail_match_serial():
    rng = np.random.default_rng(37)
    for n in (137, 100_003):
        bins = rng.integers(-1, 7, size=(n, 2), dtype=np.int64)
        grad = np.ascontiguousarray(rng.normal(size=n))
        hess = np.ascontiguousarray(rng.uniform(0.5, 1.5, size=n))
        serial = _build_random_tree(
            bins, grad, hess, max_bin=7, max_depth=2, n_threads=1
        )
        threaded = _build_random_tree(
            bins, grad, hess, max_bin=7, max_depth=2, n_threads=4
        )
        for actual, expected in zip(threaded, serial):
            assert np.array_equal(actual, expected)


def test_simd_root_tail_and_histogram_child_totals():
    rng = np.random.default_rng(39)
    n, d, max_bin = 139, 5, 7
    bins = rng.integers(-1, max_bin, size=(n, d), dtype=np.int64)
    grad = np.ascontiguousarray(bins[:, 0].astype(np.float64) - 2.0)
    hess = np.ascontiguousarray(rng.uniform(0.5, 1.5, size=n))
    built = _build_random_tree(
        bins, grad, hess, max_bin=max_bin, max_depth=2, n_threads=1
    )
    features, split_bins, defaults = built[:3]
    covers, node_grad, node_hess = built[5], built[7], built[8]
    assert features[0] >= 0
    feature = features[0]
    column = bins[:, feature]
    left = (column <= split_bins[0]) & (column >= 0)
    if defaults[0]:
        left |= column < 0
    assert np.isclose(covers[0], hess.sum(), rtol=1e-14, atol=1e-14)
    assert np.isclose(node_grad[1], grad[left].sum(), rtol=1e-14, atol=1e-14)
    assert np.isclose(node_hess[1], hess[left].sum(), rtol=1e-14, atol=1e-14)
    assert np.isclose(node_grad[2], grad[~left].sum(), rtol=1e-14, atol=1e-14)
    assert np.isclose(node_hess[2], hess[~left].sum(), rtol=1e-14, atol=1e-14)


def test_parallel_prediction_threshold_matches_serial():
    rng = np.random.default_rng(41)
    train_x = rng.normal(size=(800, 4))
    train_y = 2.0 * train_x[:, 0] - train_x[:, 1]
    model = xgb.XGBRegressor(
        n_estimators=5, max_depth=3, max_bin=17, n_jobs=1
    ).fit(train_x, train_y, verbose=False).get_booster()
    for n in (19, 20_001):
        data = rng.normal(size=(n, 4))
        model.params["n_jobs"] = 1
        serial = model.predict(xgb.DMatrix(data))
        model.params["n_jobs"] = 4
        threaded = model.predict(xgb.DMatrix(data))
        assert np.array_equal(threaded, serial)


def test_gpu_headroom_and_allocation_limits(monkeypatch):
    class Result:
        returncode = 0
        stdout = "3999\n"

    monkeypatch.setattr(_gpu.subprocess, "run", lambda *args, **kwargs: Result())
    assert not _gpu._has_headroom(1)
    Result.stdout = "4000\n"
    assert _gpu._has_headroom(_gpu.MAX_DEVICE_BYTES - 1)
    assert not _gpu._has_headroom(_gpu.MAX_DEVICE_BYTES)
