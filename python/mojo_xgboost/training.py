"""Histogram gradient boosting orchestration around the Mojo kernels."""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Iterable

import numpy as np

from ._lib import addr, lib
from .core import Booster, DMatrix, _transform


def _value(params: dict[str, Any], name: str, alias: str, default: Any) -> Any:
    return params[name] if name in params else params.get(alias, default)


def _cuts(x: np.ndarray, max_bin: int) -> np.ndarray:
    result = np.full((x.shape[1], max_bin - 1), np.inf, dtype=np.float64)
    probabilities = np.arange(1, max_bin, dtype=np.float64) / max_bin
    for feature in range(x.shape[1]):
        values = x[:, feature]
        values = values[~np.isnan(values)]
        if not values.size:
            continue
        quantiles = np.unique(np.quantile(values, probabilities, method="inverted_cdf"))
        result[feature, : quantiles.size] = quantiles
    return result


def _quantize(x: np.ndarray, cuts: np.ndarray, max_bin: int) -> np.ndarray:
    bins = np.empty(x.shape, dtype=np.int64)
    if not x.size:
        return bins
    lib().mxgb_quantize(
        addr(x), addr(cuts), addr(bins), x.shape[0], x.shape[1], max_bin
    )
    return bins


def _gradient(
    objective: str,
    margin: np.ndarray,
    label: np.ndarray,
    weight: np.ndarray,
    scale_pos_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if objective == "reg:squarederror":
        grad = margin - label
        hess = np.ones_like(grad)
    elif objective == "binary:logistic":
        pred = _transform(objective, margin)
        grad = pred - label
        hess = np.maximum(pred * (1.0 - pred), 1e-16)
        if scale_pos_weight != 1.0:
            scale = np.where(label == 1.0, scale_pos_weight, 1.0)
            grad *= scale
            hess *= scale
    else:
        raise NotImplementedError(
            "supported objectives are 'reg:squarederror' and 'binary:logistic'"
        )
    if weight.size:
        grad *= weight
        hess *= weight
    return np.ascontiguousarray(grad), np.ascontiguousarray(hess)


def _validate(params: dict[str, Any]) -> None:
    if params.get("booster", "gbtree") != "gbtree":
        raise NotImplementedError("only booster='gbtree' is implemented")
    if params.get("tree_method", "hist") not in {"hist", "auto"}:
        raise NotImplementedError("only tree_method='hist' is implemented")
    for name in ("monotone_constraints", "interaction_constraints"):
        if params.get(name) not in (None, "", (), []):
            raise NotImplementedError(f"{name} is not implemented")
    if float(params.get("colsample_bylevel", 1.0)) != 1.0:
        raise NotImplementedError("colsample_bylevel is not implemented")
    if float(params.get("colsample_bynode", 1.0)) != 1.0:
        raise NotImplementedError("colsample_bynode is not implemented")


def _metric(objective: str, name: str, label: np.ndarray, pred: np.ndarray) -> float:
    if name in {"rmse", "l2"}:
        return float(np.sqrt(np.mean((pred - label) ** 2)))
    if name in {"mae", "l1"}:
        return float(np.mean(np.abs(pred - label)))
    if name in {"logloss", "log_loss"}:
        clipped = np.clip(pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(label * np.log(clipped) + (1.0 - label) * np.log(1.0 - clipped)))
    if name in {"error", "merror"}:
        return float(np.mean((pred >= 0.5) != label))
    raise NotImplementedError(f"evaluation metric {name!r} is not implemented")


def train(
    params: dict[str, Any],
    dtrain: DMatrix,
    num_boost_round: int = 10,
    *,
    evals: Iterable[tuple[DMatrix, str]] | None = None,
    obj: Callable[[np.ndarray, DMatrix], tuple[np.ndarray, np.ndarray]] | None = None,
    maximize: bool | None = None,
    early_stopping_rounds: int | None = None,
    evals_result: dict[str, dict[str, list[float]]] | None = None,
    verbose_eval: bool | int | None = True,
    xgb_model: Booster | str | None = None,
    callbacks: Any | None = None,
    custom_metric: Callable[[np.ndarray, DMatrix], tuple[str, float]] | None = None,
) -> Booster:
    if not isinstance(dtrain, DMatrix):
        raise TypeError("dtrain must be a DMatrix")
    if not dtrain.num_row():
        raise ValueError("training data must contain at least one row")
    if not dtrain._label.size:
        raise ValueError("training labels are required")
    if callbacks:
        raise NotImplementedError("training callbacks are not implemented")
    if num_boost_round < 0:
        raise ValueError("num_boost_round must be non-negative")
    params = dict(params)
    _validate(params)
    objective = params.get("objective", "reg:squarederror")
    if objective == "binary:logistic":
        labels = np.unique(dtrain._label)
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("binary:logistic labels must be 0 or 1")
    max_depth = int(params.get("max_depth", 6))
    max_bin = int(params.get("max_bin", 256))
    if not 1 <= max_depth <= 12:
        raise ValueError("max_depth must be between 1 and 12")
    if not 2 <= max_bin <= 4096:
        raise ValueError("max_bin must be between 2 and 4096")
    eta = float(_value(params, "eta", "learning_rate", 0.3))
    min_child_weight = float(params.get("min_child_weight", 1.0))
    reg_lambda = float(_value(params, "lambda", "reg_lambda", 1.0))
    reg_alpha = float(_value(params, "alpha", "reg_alpha", 0.0))
    gamma = float(_value(params, "gamma", "min_split_loss", 0.0))
    max_delta_step = float(params.get("max_delta_step", 0.0))
    scale_pos_weight = float(params.get("scale_pos_weight", 1.0))
    numeric_params = {
        "learning_rate": eta,
        "min_child_weight": min_child_weight,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "gamma": gamma,
        "max_delta_step": max_delta_step,
        "scale_pos_weight": scale_pos_weight,
    }
    if not all(math.isfinite(value) for value in numeric_params.values()):
        raise ValueError("numeric training parameters must be finite")
    if eta < 0.0:
        raise ValueError("learning_rate must be non-negative")
    if any(value < 0.0 for value in (min_child_weight, reg_lambda, reg_alpha, gamma, max_delta_step)):
        raise ValueError("regularization and child-weight parameters must be non-negative")
    if scale_pos_weight <= 0.0:
        raise ValueError("scale_pos_weight must be positive")
    subsample = float(params.get("subsample", 1.0))
    colsample = float(params.get("colsample_bytree", 1.0))
    if not 0.0 < subsample <= 1.0 or not 0.0 < colsample <= 1.0:
        raise ValueError("subsample and colsample_bytree must be in (0, 1]")
    seed = int(_value(params, "seed", "random_state", 0))
    rng = np.random.default_rng(seed)

    if xgb_model is None:
        model = Booster(params)
    elif isinstance(xgb_model, Booster):
        model = copy.deepcopy(xgb_model)
        if model.objective != objective:
            raise ValueError("xgb_model objective does not match params")
    else:
        model = Booster(model_file=xgb_model)
    model.feature_names = list(dtrain.feature_names)
    model.params.update(params)
    model.objective = objective

    cuts = _cuts(dtrain.data, max_bin)
    all_bins = _quantize(dtrain.data, cuts, max_bin)
    n, d = all_bins.shape
    max_nodes = (1 << (max_depth + 1)) - 1
    if model._max_nodes and model._max_nodes != max_nodes:
        raise ValueError("continued training cannot change max_depth")
    histogram_size = max_nodes * d * max_bin
    hist_grad = np.empty(histogram_size, dtype=np.float64)
    hist_hess = np.empty(histogram_size, dtype=np.float64)
    row_node = np.empty(n, dtype=np.int64)
    node_grad = np.empty(max_nodes, dtype=np.float64)
    node_hess = np.empty(max_nodes, dtype=np.float64)

    evaluations = list(evals or [])
    history = evals_result if evals_result is not None else {}
    metric_names = params.get(
        "eval_metric", "logloss" if objective == "binary:logistic" else "rmse"
    )
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    for _, name in evaluations:
        history.setdefault(name, {})
        for metric_name in metric_names:
            history[name].setdefault(metric_name, [])
    no_improvement = 0
    best = -math.inf if maximize else math.inf
    n_threads = model._n_threads()
    if model.num_boosted_rounds():
        margin = model.predict(dtrain, output_margin=True)
    elif dtrain._base_margin.size:
        margin = dtrain._base_margin.copy()
    else:
        margin = np.full(n, model.base_margin, dtype=np.float64)

    for round_index in range(num_boost_round):
        if obj is None:
            grad, hess = _gradient(
                objective,
                margin,
                dtrain._label,
                dtrain._weight,
                scale_pos_weight,
            )
        else:
            grad, hess = obj(margin, dtrain)
            grad = np.ascontiguousarray(grad, dtype=np.float64)
            hess = np.ascontiguousarray(hess, dtype=np.float64)
            if grad.shape != (n,) or hess.shape != (n,):
                raise ValueError("custom objective must return two arrays of shape (n_rows,)")
            if not np.isfinite(grad).all() or not np.isfinite(hess).all():
                raise ValueError("custom objective gradients and hessians must be finite")
            if np.any(hess < 0.0):
                raise ValueError("custom objective hessians must be non-negative")

        if subsample < 1.0:
            selected = rng.random(n) < subsample
            grad = grad.copy()
            hess = hess.copy()
            grad[~selected] = 0.0
            hess[~selected] = 0.0
        bins = all_bins
        if colsample < 1.0:
            count = max(1, int(math.ceil(colsample * d)))
            selected_features = rng.choice(d, count, replace=False)
            bins = all_bins.copy()
            excluded = np.ones(d, dtype=bool)
            excluded[selected_features] = False
            bins[:, excluded] = -1

        features = np.empty(max_nodes, dtype=np.int64)
        split_bins = np.empty(max_nodes, dtype=np.int64)
        defaults = np.empty(max_nodes, dtype=np.int64)
        leaves = np.empty(max_nodes, dtype=np.float64)
        gains = np.empty(max_nodes, dtype=np.float64)
        covers = np.empty(max_nodes, dtype=np.float64)
        split_count = lib().mxgb_build_tree(
            addr(bins),
            addr(grad),
            addr(hess),
            addr(features),
            addr(split_bins),
            addr(defaults),
            addr(leaves),
            addr(gains),
            addr(covers),
            addr(row_node),
            addr(node_grad),
            addr(node_hess),
            addr(hist_grad),
            addr(hist_hess),
            n,
            d,
            max_bin,
            max_depth,
            min_child_weight,
            reg_lambda,
            reg_alpha,
            gamma,
            max_delta_step,
            n_threads,
        )
        if split_count < 0:
            raise RuntimeError("Mojo tree builder rejected invalid native arguments")
        thresholds = np.full(max_nodes, np.nan, dtype=np.float64)
        split_nodes = np.flatnonzero(features >= 0)
        thresholds[split_nodes] = cuts[
            features[split_nodes], split_bins[split_nodes]
        ]
        leaves *= eta
        model._append(features, thresholds, defaults, leaves, gains, covers)
        lib().mxgb_predict_add(
            addr(dtrain.data),
            addr(features),
            addr(thresholds),
            addr(defaults),
            addr(leaves),
            addr(margin),
            n,
            d,
            n_threads,
        )

        latest: float | None = None
        latest_name = ""
        messages: list[str] = []
        for matrix, eval_name in evaluations:
            prediction = model.predict(matrix)
            for metric_name in metric_names:
                value = _metric(objective, metric_name, matrix._label, prediction)
                history[eval_name][metric_name].append(value)
                latest, latest_name = value, f"{eval_name}-{metric_name}"
                messages.append(f"{latest_name}:{value:.6f}")
            if custom_metric is not None:
                custom_name, value = custom_metric(prediction, matrix)
                history[eval_name].setdefault(custom_name, []).append(float(value))
                latest, latest_name = float(value), f"{eval_name}-{custom_name}"
                messages.append(f"{latest_name}:{value:.6f}")
        if messages and verbose_eval:
            period = int(verbose_eval) if isinstance(verbose_eval, int) else 1
            if round_index % period == 0 or round_index == num_boost_round - 1:
                print(f"[{round_index}]\t" + "\t".join(messages))
        if early_stopping_rounds is not None:
            if latest is None:
                raise ValueError("early_stopping_rounds requires at least one eval set")
            improved = latest > best if maximize else latest < best
            if improved:
                best = latest
                model.best_score = latest
                model.best_iteration = model.num_boosted_rounds() - 1
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= early_stopping_rounds:
                    break
    return model
