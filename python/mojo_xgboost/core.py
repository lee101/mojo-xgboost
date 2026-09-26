"""XGBoost-compatible data and model objects for the covered dense subset."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterable

import numpy as np

from ._lib import addr, f64, i64, lib, run_row_kernel


def _matrix(data: Any, missing: float = np.nan) -> np.ndarray:
    if isinstance(data, DMatrix):
        return data.data
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"data must be two-dimensional, got shape {array.shape}")
    if not array.shape[1]:
        raise ValueError("data must contain at least one feature")
    result = f64(array, copy=not np.isnan(missing))
    if not np.isnan(missing):
        result[result == missing] = np.nan
    if np.isinf(result).any():
        raise ValueError("input contains infinity")
    return result


class DMatrix:
    """Dense numeric matrix with the commonly used upstream metadata methods."""

    def __init__(
        self,
        data: Any,
        label: Any | None = None,
        *,
        weight: Any | None = None,
        base_margin: Any | None = None,
        missing: float = np.nan,
        silent: bool = False,
        feature_names: Iterable[str] | None = None,
        feature_types: Iterable[str] | None = None,
        nthread: int | None = None,
        enable_categorical: bool = False,
        data_split_mode: Any | None = None,
    ):
        del silent, nthread, data_split_mode
        if enable_categorical:
            raise NotImplementedError("categorical features are outside this port's scope")
        self.data = _matrix(data, missing)
        self.missing = missing
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"f{i}" for i in range(self.data.shape[1])]
        )
        if len(self.feature_names) != self.data.shape[1]:
            raise ValueError("feature_names length must match the number of columns")
        self.feature_types = list(feature_types) if feature_types is not None else None
        self._label = self._metadata(label, "label")
        self._weight = self._metadata(weight, "weight")
        self._base_margin = self._metadata(base_margin, "base_margin")

    def _metadata(self, values: Any | None, name: str) -> np.ndarray:
        if values is None:
            return np.empty(0, dtype=np.float64)
        result = f64(values).reshape(-1)
        if result.size != self.data.shape[0]:
            raise ValueError(f"{name} length must equal the number of rows")
        if not np.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        if name == "weight" and np.any(result < 0.0):
            raise ValueError("weight must be non-negative")
        return result

    def num_row(self) -> int:
        return self.data.shape[0]

    def num_col(self) -> int:
        return self.data.shape[1]

    def num_nonmissing(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.data)))

    def get_label(self) -> np.ndarray:
        return self._label.copy()

    def get_weight(self) -> np.ndarray:
        return self._weight.copy()

    def get_base_margin(self) -> np.ndarray:
        return self._base_margin.copy()

    def set_label(self, label: Any) -> None:
        self._label = self._metadata(label, "label")

    def set_weight(self, weight: Any) -> None:
        self._weight = self._metadata(weight, "weight")

    def set_base_margin(self, margin: Any) -> None:
        self._base_margin = self._metadata(margin, "base_margin")

    def set_info(
        self,
        *,
        label: Any | None = None,
        weight: Any | None = None,
        base_margin: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise NotImplementedError(f"unsupported metadata: {', '.join(kwargs)}")
        if label is not None:
            self.set_label(label)
        if weight is not None:
            self.set_weight(weight)
        if base_margin is not None:
            self.set_base_margin(base_margin)

    def slice(self, rindex: Any, allow_groups: bool = False) -> "DMatrix":
        del allow_groups
        index = i64(rindex)
        return DMatrix(
            self.data[index],
            self._label[index] if self._label.size else None,
            weight=self._weight[index] if self._weight.size else None,
            base_margin=self._base_margin[index] if self._base_margin.size else None,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
        )


class Booster:
    """An ensemble of dense depth-wise histogram trees."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        cache: Iterable[DMatrix] = (),
        model_file: str | os.PathLike[str] | None = None,
    ):
        del cache
        self.params = dict(params or {})
        self.objective = self.params.get("objective", "reg:squarederror")
        self.base_score = float(self.params.get("base_score", 0.5))
        self.base_margin = _base_margin(self.objective, self.base_score)
        self.feature_names: list[str] | None = None
        self._features: list[np.ndarray] = []
        self._thresholds: list[np.ndarray] = []
        self._defaults: list[np.ndarray] = []
        self._leaves: list[np.ndarray] = []
        self._gains: list[np.ndarray] = []
        self._covers: list[np.ndarray] = []
        self._max_nodes = 0
        self.best_iteration: int | None = None
        self.best_score: float | None = None
        if model_file is not None:
            self.load_model(model_file)

    def _append(
        self,
        features: np.ndarray,
        thresholds: np.ndarray,
        defaults: np.ndarray,
        leaves: np.ndarray,
        gains: np.ndarray,
        covers: np.ndarray,
    ) -> None:
        if self._max_nodes and features.size != self._max_nodes:
            raise ValueError("all trees in a booster must use the same max_depth")
        self._max_nodes = features.size
        self._features.append(i64(features))
        self._thresholds.append(f64(thresholds))
        self._defaults.append(i64(defaults))
        self._leaves.append(f64(leaves))
        self._gains.append(f64(gains))
        self._covers.append(f64(covers))

    def num_boosted_rounds(self) -> int:
        return len(self._features)

    def num_features(self) -> int:
        return len(self.feature_names or [])

    def _tree_range(self, iteration_range: tuple[int, int]) -> tuple[int, int]:
        start, end = iteration_range
        if end == 0:
            end = len(self._features)
        if start < 0 or end < start or end > len(self._features):
            raise ValueError("invalid iteration_range")
        return start, end

    def _n_threads(self) -> int:
        value = self.params.get("n_jobs")
        if value is None:
            return 1
        threads = int(value)
        if threads < 0:
            return os.cpu_count() or 1
        return max(1, threads)

    def predict(
        self,
        data: DMatrix,
        *,
        output_margin: bool = False,
        pred_leaf: bool = False,
        pred_contribs: bool = False,
        approx_contribs: bool = False,
        pred_interactions: bool = False,
        validate_features: bool = True,
        training: bool = False,
        iteration_range: tuple[int, int] = (0, 0),
        strict_shape: bool = False,
    ) -> np.ndarray:
        del approx_contribs, training, strict_shape
        if pred_contribs or pred_interactions:
            raise NotImplementedError("SHAP contribution outputs are not implemented")
        matrix = data if isinstance(data, DMatrix) else DMatrix(data)
        if validate_features and self.feature_names is not None:
            if matrix.num_col() != len(self.feature_names):
                raise ValueError("feature count does not match the trained booster")
        start, end = self._tree_range(iteration_range)
        n_trees = end - start
        if pred_leaf:
            result = np.empty((matrix.num_row(), n_trees), dtype=np.int64)
            if n_trees:
                features, thresholds, defaults, _ = self._stack(start, end)
                run_row_kernel(
                    "mxgb_predict_leaf_range",
                    (
                        addr(matrix.data),
                        addr(features),
                        addr(thresholds),
                        addr(defaults),
                        addr(result),
                        matrix.num_row(),
                        matrix.num_col(),
                        n_trees,
                        self._max_nodes,
                    ),
                    matrix.num_row(),
                    self._n_threads(),
                    matrix.num_row() * n_trees,
                )
            return result
        result = np.empty(matrix.num_row(), dtype=np.float64)
        margin = self.base_margin
        if matrix._base_margin.size:
            margin = 0.0
        if n_trees:
            features, thresholds, defaults, leaves = self._stack(start, end)
            run_row_kernel(
                "mxgb_predict_range",
                (
                    addr(matrix.data),
                    addr(features),
                    addr(thresholds),
                    addr(defaults),
                    addr(leaves),
                    addr(result),
                    matrix.num_row(),
                    matrix.num_col(),
                    n_trees,
                    self._max_nodes,
                    margin,
                ),
                matrix.num_row(),
                self._n_threads(),
                matrix.num_row() * n_trees,
            )
        else:
            result.fill(margin)
        if matrix._base_margin.size:
            result += matrix._base_margin
        if output_margin:
            return result
        return _transform(self.objective, result)

    def inplace_predict(
        self,
        data: Any,
        *,
        iteration_range: tuple[int, int] = (0, 0),
        predict_type: str = "value",
        missing: float = np.nan,
        validate_features: bool = True,
        base_margin: Any | None = None,
        strict_shape: bool = False,
    ) -> np.ndarray:
        matrix = DMatrix(data, missing=missing, base_margin=base_margin)
        return self.predict(
            matrix,
            output_margin=predict_type == "margin",
            validate_features=validate_features,
            iteration_range=iteration_range,
            strict_shape=strict_shape,
        )

    def _stack(
        self, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.ascontiguousarray(self._features[start:end]),
            np.ascontiguousarray(self._thresholds[start:end]),
            np.ascontiguousarray(self._defaults[start:end]),
            np.ascontiguousarray(self._leaves[start:end]),
        )

    def get_score(self, fmap: str = "", importance_type: str = "weight") -> dict[str, float]:
        del fmap
        valid = {"weight", "gain", "cover", "total_gain", "total_cover"}
        if importance_type not in valid:
            raise ValueError(f"importance_type must be one of {sorted(valid)}")
        totals: dict[int, list[float]] = {}
        for features, gains, covers in zip(self._features, self._gains, self._covers):
            for node in np.flatnonzero(features >= 0):
                feature = int(features[node])
                values = totals.setdefault(feature, [0.0, 0.0, 0.0])
                values[0] += 1.0
                values[1] += float(gains[node])
                values[2] += float(covers[node])
        result: dict[str, float] = {}
        names = self.feature_names or [f"f{i}" for i in range(self.num_features())]
        for feature, (count, gain, cover) in totals.items():
            if importance_type == "weight":
                value = count
            elif importance_type == "gain":
                value = gain / count
            elif importance_type == "cover":
                value = cover / count
            elif importance_type == "total_gain":
                value = gain
            else:
                value = cover
            result[names[feature]] = value
        return result

    def get_dump(
        self, fmap: str = "", with_stats: bool = False, dump_format: str = "text"
    ) -> list[str]:
        del fmap
        if dump_format not in {"text", "json"}:
            raise ValueError("dump_format must be 'text' or 'json'")
        return [
            self._dump_tree(i, with_stats, dump_format)
            for i in range(len(self._features))
        ]

    def _dump_tree(self, tree: int, stats: bool, format_: str) -> str:
        features = self._features[tree]
        if format_ == "json":
            def node_json(node: int) -> dict[str, Any]:
                if features[node] < 0:
                    value: dict[str, Any] = {"nodeid": node, "leaf": self._leaves[tree][node]}
                else:
                    f = int(features[node])
                    value = {
                        "nodeid": node,
                        "split": (self.feature_names or [])[f] if self.feature_names else f"f{f}",
                        "split_condition": self._thresholds[tree][node],
                        "yes": 2 * node + 1,
                        "no": 2 * node + 2,
                        "missing": 2 * node + (1 if self._defaults[tree][node] else 2),
                        "children": [node_json(2 * node + 1), node_json(2 * node + 2)],
                    }
                if stats:
                    value["cover"] = self._covers[tree][node]
                    value["gain"] = self._gains[tree][node]
                return value
            return json.dumps(node_json(0))
        lines: list[str] = []
        def visit(node: int, depth: int) -> None:
            prefix = "\t" * depth
            if features[node] < 0:
                lines.append(f"{prefix}{node}:leaf={self._leaves[tree][node]:.9g}")
                return
            f = int(features[node])
            name = self.feature_names[f] if self.feature_names else f"f{f}"
            yes, no = 2 * node + 1, 2 * node + 2
            missing = yes if self._defaults[tree][node] else no
            suffix = (
                f",gain={self._gains[tree][node]:.9g},cover={self._covers[tree][node]:.9g}"
                if stats else ""
            )
            lines.append(
                f"{prefix}{node}:[{name}<={self._thresholds[tree][node]:.9g}] "
                f"yes={yes},no={no},missing={missing}{suffix}"
            )
            visit(yes, depth + 1)
            visit(no, depth + 1)
        visit(0, 0)
        return "\n".join(lines)

    def save_model(self, fname: str | os.PathLike[str]) -> None:
        state = {
            "format": "mojo-xgboost-1",
            "params": self.params,
            "objective": self.objective,
            "base_score": self.base_score,
            "base_margin": self.base_margin,
            "feature_names": self.feature_names,
            "max_nodes": self._max_nodes,
            "features": [x.tolist() for x in self._features],
            "thresholds": [x.tolist() for x in self._thresholds],
            "defaults": [x.tolist() for x in self._defaults],
            "leaves": [x.tolist() for x in self._leaves],
            "gains": [x.tolist() for x in self._gains],
            "covers": [x.tolist() for x in self._covers],
        }
        with open(fname, "w", encoding="utf-8") as handle:
            json.dump(state, handle, allow_nan=True)

    def load_model(self, fname: str | os.PathLike[str]) -> None:
        with open(fname, encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("format") != "mojo-xgboost-1":
            raise ValueError("not a mojo-xgboost model")
        required = ("features", "thresholds", "defaults", "leaves", "gains", "covers")
        arrays = {name: state.get(name) for name in required}
        if any(not isinstance(arrays[name], list) for name in required):
            raise ValueError("model tree arrays must be lists")
        tree_count = len(arrays["features"])
        if any(len(arrays[name]) != tree_count for name in required):
            raise ValueError("model tree array counts do not match")
        max_nodes = state.get("max_nodes")
        if not isinstance(max_nodes, int) or max_nodes < 0:
            raise ValueError("invalid max_nodes")
        converted: dict[str, list[np.ndarray]] = {
            "features": [i64(x) for x in arrays["features"]],
            "thresholds": [f64(x) for x in arrays["thresholds"]],
            "defaults": [i64(x) for x in arrays["defaults"]],
            "leaves": [f64(x) for x in arrays["leaves"]],
            "gains": [f64(x) for x in arrays["gains"]],
            "covers": [f64(x) for x in arrays["covers"]],
        }
        if any(
            array.ndim != 1 or array.size != max_nodes
            for values in converted.values()
            for array in values
        ):
            raise ValueError("model tree arrays have invalid lengths")
        feature_names = state.get("feature_names")
        if feature_names is not None and (
            not isinstance(feature_names, list)
            or not all(isinstance(name, str) for name in feature_names)
        ):
            raise ValueError("invalid feature_names")
        feature_count = len(feature_names or [])
        if tree_count and not feature_count:
            raise ValueError("tree models require feature_names")
        for features, thresholds, defaults in zip(
            converted["features"], converted["thresholds"], converted["defaults"]
        ):
            if np.any(features < -1):
                raise ValueError("invalid model feature index")
            if feature_count and np.any(features >= feature_count):
                raise ValueError("model feature index is out of bounds")
            if max_nodes and np.any(features[(max_nodes - 1) // 2 :] >= 0):
                raise ValueError("model splits below maximum tree depth")
            if np.any(~np.isfinite(thresholds[features >= 0])):
                raise ValueError("model split thresholds must be finite")
            if np.any((defaults != 0) & (defaults != 1)):
                raise ValueError("invalid model missing direction")
        objective = state.get("objective")
        if objective not in {"reg:squarederror", "binary:logistic"}:
            raise ValueError("unsupported model objective")
        self.params = dict(state.get("params", {}))
        self.objective = objective
        self.base_score = float(state["base_score"])
        self.base_margin = float(state["base_margin"])
        if not math.isfinite(self.base_score) or not math.isfinite(self.base_margin):
            raise ValueError("model base values must be finite")
        self.feature_names = feature_names
        self._features = converted["features"]
        self._thresholds = converted["thresholds"]
        self._defaults = converted["defaults"]
        self._leaves = converted["leaves"]
        self._gains = converted["gains"]
        self._covers = converted["covers"]
        self._max_nodes = max_nodes

    def save_raw(self, raw_format: str = "json") -> bytearray:
        if raw_format != "json":
            raise ValueError("only raw_format='json' is supported")
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as handle:
            self.save_model(handle.name)
            handle.seek(0)
            return bytearray(handle.read().encode())


def _base_margin(objective: str, base_score: float) -> float:
    if objective == "binary:logistic":
        if not 0.0 < base_score < 1.0:
            raise ValueError("base_score must be in (0, 1) for binary:logistic")
        return math.log(base_score / (1.0 - base_score))
    return base_score


def _transform(objective: str, margin: np.ndarray) -> np.ndarray:
    if objective == "binary:logistic":
        result = np.empty_like(margin)
        positive = margin >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-margin[positive]))
        exp_margin = np.exp(margin[~positive])
        result[~positive] = exp_margin / (1.0 + exp_margin)
        return result
    return margin
