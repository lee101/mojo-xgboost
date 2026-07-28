"""Scikit-learn-style estimators with XGBoost parameter names."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import Booster, DMatrix
from .training import train


class XGBModel:
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 6,
        max_leaves: int = 0,
        max_bin: int = 256,
        learning_rate: float = 0.3,
        verbosity: int = 1,
        objective: str | None = None,
        booster: str = "gbtree",
        tree_method: str = "hist",
        n_jobs: int | None = None,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        max_delta_step: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        colsample_bynode: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        base_score: float = 0.5,
        random_state: int | None = None,
        missing: float = np.nan,
        importance_type: str | None = None,
        eval_metric: str | list[str] | None = None,
        early_stopping_rounds: int | None = None,
        callbacks: Any | None = None,
        **kwargs: Any,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.objective = objective
        self.booster = booster
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.missing = missing
        self.importance_type = importance_type
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        self.kwargs = dict(kwargs)
        self._Booster: Booster | None = None
        self.evals_result_: dict[str, dict[str, list[float]]] = {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        names = (
            "n_estimators", "max_depth", "max_leaves", "max_bin", "learning_rate",
            "verbosity", "objective", "booster", "tree_method", "n_jobs", "gamma",
            "min_child_weight", "max_delta_step", "subsample", "colsample_bytree",
            "colsample_bylevel", "colsample_bynode", "reg_alpha", "reg_lambda",
            "scale_pos_weight", "base_score", "random_state", "missing",
            "importance_type", "eval_metric", "early_stopping_rounds", "callbacks",
        )
        result = {name: getattr(self, name) for name in names}
        result.update(self.kwargs)
        return result

    def set_params(self, **params: Any) -> "XGBModel":
        for name, value in params.items():
            if hasattr(self, name) and name != "kwargs":
                setattr(self, name, value)
            else:
                self.kwargs[name] = value
        return self

    def _params(self) -> dict[str, Any]:
        params = self.get_params()
        params.pop("n_estimators")
        params["n_jobs"] = self.n_jobs
        params.pop("verbosity")
        params.pop("missing")
        params.pop("importance_type")
        params.pop("early_stopping_rounds")
        params.pop("callbacks")
        if params.pop("max_leaves", 0):
            raise NotImplementedError("max_leaves/lossguide growth is not implemented")
        if params["objective"] is None:
            params["objective"] = self._default_objective
        if params["eval_metric"] is None:
            params.pop("eval_metric")
        params["seed"] = 0 if self.random_state is None else self.random_state
        return params

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any | None = None,
        base_margin: Any | None = None,
        eval_set: list[tuple[Any, Any]] | None = None,
        verbose: bool | int | None = True,
        xgb_model: Booster | str | None = None,
        sample_weight_eval_set: list[Any] | None = None,
        base_margin_eval_set: list[Any] | None = None,
        feature_weights: Any | None = None,
    ) -> "XGBModel":
        if feature_weights is not None:
            raise NotImplementedError("feature_weights is not implemented")
        training = DMatrix(
            X, y, weight=sample_weight, base_margin=base_margin, missing=self.missing
        )
        evaluations = []
        for index, (eval_x, eval_y) in enumerate(eval_set or []):
            eval_weight = (
                sample_weight_eval_set[index] if sample_weight_eval_set else None
            )
            eval_margin = (
                base_margin_eval_set[index] if base_margin_eval_set else None
            )
            evaluations.append(
                (
                    DMatrix(
                        eval_x,
                        eval_y,
                        weight=eval_weight,
                        base_margin=eval_margin,
                        missing=self.missing,
                    ),
                    f"validation_{index}",
                )
            )
        self.evals_result_ = {}
        self._Booster = train(
            self._params(),
            training,
            self.n_estimators,
            evals=evaluations,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self.evals_result_,
            verbose_eval=verbose,
            xgb_model=xgb_model,
            callbacks=self.callbacks,
        )
        self.n_features_in_ = training.num_col()
        return self

    def get_booster(self) -> Booster:
        if self._Booster is None:
            raise ValueError("estimator is not fitted")
        return self._Booster

    def evals_result(self) -> dict[str, dict[str, list[float]]]:
        if self._Booster is None:
            raise ValueError("estimator is not fitted")
        return self.evals_result_

    def apply(self, X: Any, iteration_range: tuple[int, int] = (0, 0)) -> np.ndarray:
        return self.get_booster().predict(
            DMatrix(X, missing=self.missing),
            pred_leaf=True,
            iteration_range=iteration_range,
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        booster = self.get_booster()
        scores = booster.get_score(importance_type=self.importance_type or "gain")
        result = np.array(
            [scores.get(f"f{i}", 0.0) for i in range(self.n_features_in_)],
            dtype=np.float64,
        )
        total = result.sum()
        return result / total if total else result

    @property
    def best_iteration(self) -> int:
        booster = self.get_booster()
        return (
            booster.best_iteration
            if booster.best_iteration is not None
            else booster.num_boosted_rounds() - 1
        )

    @property
    def best_score(self) -> float:
        score = self.get_booster().best_score
        if score is None:
            raise AttributeError("best_score is defined only with early stopping")
        return score


class XGBRegressor(XGBModel):
    _default_objective = "reg:squarederror"

    def __init__(self, *, objective: str = "reg:squarederror", **kwargs: Any):
        super().__init__(objective=objective, **kwargs)

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Any | None = None,
        iteration_range: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        return self.get_booster().predict(
            DMatrix(X, base_margin=base_margin, missing=self.missing),
            output_margin=output_margin,
            validate_features=validate_features,
            iteration_range=iteration_range,
        )

    def score(self, X: Any, y: Any) -> float:
        actual = np.asarray(y, dtype=np.float64)
        residual = np.sum((actual - self.predict(X)) ** 2)
        total = np.sum((actual - actual.mean()) ** 2)
        return float(1.0 - residual / total)


class XGBClassifier(XGBModel):
    _default_objective = "binary:logistic"

    def __init__(self, *, objective: str = "binary:logistic", **kwargs: Any):
        super().__init__(objective=objective, **kwargs)

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "XGBClassifier":
        labels = np.asarray(y)
        classes = np.unique(labels)
        if classes.size != 2 or not np.array_equal(classes, [0, 1]):
            raise ValueError("XGBClassifier currently requires labels 0 and 1")
        self.classes_ = classes
        super().fit(X, labels, **kwargs)
        return self

    def predict_proba(
        self,
        X: Any,
        *,
        validate_features: bool = True,
        base_margin: Any | None = None,
        iteration_range: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        positive = self.get_booster().predict(
            DMatrix(X, base_margin=base_margin, missing=self.missing),
            validate_features=validate_features,
            iteration_range=iteration_range,
        )
        return np.column_stack((1.0 - positive, positive))

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        validate_features: bool = True,
        base_margin: Any | None = None,
        iteration_range: tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        matrix = DMatrix(X, base_margin=base_margin, missing=self.missing)
        values = self.get_booster().predict(
            matrix,
            output_margin=output_margin,
            validate_features=validate_features,
            iteration_range=iteration_range,
        )
        return values if output_margin else (values >= 0.5).astype(np.int64)

    def score(self, X: Any, y: Any) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
