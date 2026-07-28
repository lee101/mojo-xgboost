"""Measured mojo-xgboost versus upstream xgboost on identical dense data."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import xgboost as upstream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_xgboost as mxgb  # noqa: E402


def best_time(function, repetitions: int) -> tuple[float, object]:
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def row(name: str, mojo_seconds: float, upstream_seconds: float) -> None:
    speedup = upstream_seconds / mojo_seconds
    print(
        f"| {name} | {mojo_seconds * 1000:.2f} ms | "
        f"{upstream_seconds * 1000:.2f} ms | {speedup:.2f}x |"
    )


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(21)
    n, d = 50_000, 16
    x = rng.normal(size=(n, d))
    coefficients = rng.normal(size=d)
    y_regression = x @ coefficients + rng.normal(scale=0.5, size=n)
    y_binary = (x[:, :6].sum(axis=1) + rng.normal(size=n) > 0.0).astype(np.int64)
    common = dict(
        n_estimators=30,
        max_depth=4,
        max_bin=64,
        learning_rate=0.1,
        tree_method="hist",
        base_score=0.5,
        random_state=7,
        n_jobs=1,
        verbosity=0,
    )

    mojo_regression = lambda: mxgb.XGBRegressor(**common).fit(
        x, y_regression, verbose=False
    )
    upstream_regression = lambda: upstream.XGBRegressor(**common).fit(x, y_regression)
    mojo_fit, mojo_model = best_time(mojo_regression, 3)
    upstream_fit, upstream_model = best_time(upstream_regression, 3)

    prediction_x = np.ascontiguousarray(rng.normal(size=(200_000, d)))
    mojo_predict, _ = best_time(lambda: mojo_model.predict(prediction_x), 5)
    upstream_predict, _ = best_time(lambda: upstream_model.predict(prediction_x), 5)

    mojo_classification = lambda: mxgb.XGBClassifier(**common).fit(
        x, y_binary, verbose=False
    )
    upstream_classification = lambda: upstream.XGBClassifier(**common).fit(x, y_binary)
    mojo_classify_fit, _ = best_time(mojo_classification, 3)
    upstream_classify_fit, _ = best_time(upstream_classification, 3)

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}")
    print(f"Versions: Mojo 1.0.0 nightly, xgboost {upstream.__version__}, NumPy {np.__version__}")
    print("Both implementations use one CPU thread; best of 3 fits and 5 predictions.")
    print()
    print("| Workload | mojo-xgboost | upstream xgboost | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    row("regression fit, 50k x 16, 30 trees", mojo_fit, upstream_fit)
    row("binary fit, 50k x 16, 30 trees", mojo_classify_fit, upstream_classify_fit)
    row("regression predict, 200k x 16, 30 trees", mojo_predict, upstream_predict)


if __name__ == "__main__":
    main()
