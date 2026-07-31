# mojo-xgboost

`mojo-xgboost` is a standalone Mojo implementation of the compute-heavy core of
dense histogram gradient boosting. Its Python package follows the familiar
XGBoost API: use `DMatrix` and `train`, or the scikit-learn-style
`XGBRegressor` and `XGBClassifier`.

This is an independent port, not a binding to `libxgboost` and not an official
XGBoost project. XGBoost 3.3.0 is a test and benchmark dependency so the covered
behavior can be checked against the real upstream implementation.

## Coverage

Implemented:

- dense, row-major numeric input with learned missing-value directions;
- depth-wise `gbtree` growth with `tree_method="hist"`;
- quantile cuts, binning, gradient/hessian histograms, regularized split gain,
  row partitioning, and ensemble traversal;
- `reg:squarederror` and `binary:logistic`;
- learning rate, `gamma`, `min_child_weight`, L1/L2 regularization,
  `max_delta_step`, `scale_pos_weight`, row subsampling, and per-tree column
  sampling;
- thresholded CPU parallelism for large tree-building and prediction workloads,
  controlled by `n_jobs`;
- `DMatrix`, `train`, `Booster`, `XGBModel`, `XGBRegressor`, and
  `XGBClassifier`, including evaluation history, early stopping, custom
  objectives, leaf indices, feature importance, tree dumps, and JSON model
  persistence.

Not implemented:

- sparse CSR/CSC data, categorical features, weighted quantile sketches, or
  external-memory matrices;
- multiclass, ranking, survival, count, and custom prediction transforms;
- `dart`, `gblinear`, exact/approx tree methods, loss-guide growth, monotonic or
  interaction constraints;
- distributed execution;
- SHAP contributions or compatibility with upstream XGBoost model files.

The public names and commonly used signatures match upstream for this subset,
but floating-point predictions are not expected to be identical on arbitrary
data. The quantile sketch and several tie-breaking details differ. Tests assert
exact results where the algorithms coincide and held-out numerical/behavioral
parity elsewhere.

## GPU

`src/gpu.mojo` builds to a second shared library, `dist/libmojo-xgboost-gpu.so`,
and covers the three stages that are wide, regular and independent per element:

| stage | shape | CPU | GPU (Mojo) | speedup |
|---|---|---|---|---|
| quantize | 100k x 32 | 104.8 ms | 21.3 ms | 4.9x |
| quantize | 2M x 64 | 4543.3 ms | 2319.1 ms | 2.0x |
| histogram | 100k x 32 | 43.6 ms | 5.4 ms | 8.1x |
| histogram | 500k x 64 | 1102.4 ms | 42.6 ms | 25.9x |
| histogram | 2M x 64 | 4645.2 ms | 169.5 ms | 27.4x |
| predict | 500k rows x 200 trees | 6481.0 ms | 34.6 ms | 187x |

RTX 5090, `bench/bench_gpu.py`, timings **include** the host-device copies because
that is what a caller pays. CPU is this project's own Mojo SIMD kernel for quantize
and predict, and NumPy `bincount` for the histogram (`np.add.at` would have made
the GPU look ~50x better than it is). Quantize is transfer-bound — a binary search
over a cached cut vector is only a few ALU ops per value — which is exactly why the
histogram and prediction kernels, which reuse data on the device, win by so much
more.

Design notes:

- the histogram reads bins **feature-major** (`bins_t[f * n + r]`) so adjacent
  threads touch adjacent addresses; `mxgb_gpu_quantize` writes both layouts in one
  pass, making the transpose free;
- each block accumulates into a shared-memory histogram and commits once with
  global atomics, so global traffic is `blocks * max_bin` rather than `n * d`;
- `max_bin <= 256` on the GPU path (the shared histogram is 2 x 256 x 8 B = 4 KiB
  per block); larger values fall back to the CPU;
- below ~2M cells the PCIe round trip costs more than the kernel saves, so the
  bridge declines and the CPU kernel runs. The thresholds are in `python/mojo_xgboost/_gpu.py`.

Availability is re-checked at call time, never assumed: another process holding the
card makes `DeviceContext()` fail with OOM, and every entry point then returns
`None` so the CPU path runs. `MOJO_XGBOOST_DISABLE_GPU=1` forces that. Parity with
the CPU kernels is asserted in `tests/test_gpu.py` — quantize and predict are exact,
the histogram matches `bincount` to 1e-9 (float reassociation only).

## Install

The repository pins the Mojo nightly used to build it and obtains all other
dependencies from conda-forge:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-xgboost.so`. The test suite checks native
kernels against direct calculations and exercises held-out behavior against
upstream XGBoost.

## Usage

```python
import numpy as np
import mojo_xgboost as xgb

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 6))
y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=1000)

model = xgb.XGBRegressor(
    n_estimators=40,
    max_depth=4,
    max_bin=64,
    learning_rate=0.1,
    tree_method="hist",
).fit(X, y, verbose=False)

prediction = model.predict(X[:5])
print(prediction)

dtrain = xgb.DMatrix(X, label=y)
booster = xgb.train(
    {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": 4},
    dtrain,
    num_boost_round=20,
    verbose_eval=False,
)
print(booster.predict(xgb.DMatrix(X[:5])))
```

The same program is checked into `examples/basic.py`; run it from the repository
with `pixi run python examples/basic.py`. The Pixi environment sets `PYTHONPATH`
to the repository's `python` directory.

## How it works

Python owns all model, training, and scratch arrays. Dense features are
C-contiguous `float64`; bin IDs, feature IDs, and leaf indices are contiguous
`int64`. Quantile cut selection and boosting-round orchestration live in Python.
A single Mojo compilation unit performs SIMD bin assignment, builds the
lower-cover child histogram and derives its sibling by SIMD subtraction,
evaluates both possible missing-value directions for every split, partitions
rows, and traverses multiple trees in SIMD lanes for prediction. Training
updates margins with only the newly built tree instead of repeatedly traversing
the full ensemble. Large independent feature and row workloads use thresholded
CPU parallelism; smaller inputs stay serial to avoid launch overhead.

The shared library has a small C ABI loaded with `ctypes`. Buffers cross that ABI
as integer addresses and are reconstructed as mutable `UnsafePointer` values in
Mojo. The caller owns every allocation, so the native boundary has no allocator
or lifetime coupling.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic, Mojo 1.0.0 nightly, XGBoost 3.3.0, and NumPy 2.5.1. Both
implementations were restricted to one CPU thread. Times are the best of three
fits and five predictions.

| Workload | mojo-xgboost | upstream xgboost | upstream / Mojo |
|---|---:|---:|---:|
| regression fit, 50k x 16, 30 trees | 475.08 ms | 325.56 ms | 0.69x |
| binary fit, 50k x 16, 30 trees | 702.87 ms | 482.30 ms | 0.69x |
| regression predict, 200k x 16, 30 trees | 147.90 ms | 93.67 ms | 0.63x |

These are raw timings, not a speedup claim. Lower time is better; in this run
upstream XGBoost was faster on all three workloads.
