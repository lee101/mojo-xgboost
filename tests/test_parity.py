import json

import numpy as np
import pytest
import xgboost as upstream
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split

import mojo_xgboost as xgb


def test_dmatrix_metadata_and_slice():
    x = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    matrix = xgb.DMatrix(x, y, weight=np.arange(1.0, 11.0), feature_names=["a", "b"])
    assert matrix.num_row() == 10
    assert matrix.num_col() == 2
    assert matrix.num_nonmissing() == 20
    assert np.array_equal(matrix.get_label(), y)
    sliced = matrix.slice([1, 4, 7])
    assert sliced.feature_names == ["a", "b"]
    assert np.array_equal(sliced.get_label(), y[[1, 4, 7]])
    assert np.array_equal(sliced.get_weight(), np.arange(1.0, 11.0)[[1, 4, 7]])


def test_dmatrix_keeps_contiguous_float64_input_zero_copy():
    data = np.arange(24.0).reshape(8, 3)
    matrix = xgb.DMatrix(data)
    assert matrix.data is data


@pytest.mark.parametrize(
    "data,message",
    [
        (np.array([[1.0 + 2.0j]]), "real numeric"),
        (np.array([[2**53 + 1]], dtype=np.int64), "lose float64 precision"),
        (np.empty((2, 0)), "at least one feature"),
    ],
)
def test_dmatrix_rejects_unsafe_native_conversions(data, message):
    with pytest.raises((TypeError, ValueError), match=message):
        xgb.DMatrix(data)


def test_empty_prediction_is_safe_and_empty():
    model = xgb.Booster({"objective": "reg:squarederror"})
    result = model.predict(xgb.DMatrix(np.empty((0, 3))))
    assert result.shape == (0,)


def test_training_rejects_empty_rows_and_invalid_weights():
    with pytest.raises(ValueError, match="at least one row"):
        xgb.train(
            {},
            xgb.DMatrix(np.empty((0, 2)), np.empty(0)),
            verbose_eval=False,
        )
    with pytest.raises(ValueError, match="non-negative"):
        xgb.DMatrix([[1.0]], [0.0], weight=[-1.0])


def test_missing_sentinel_is_normalized():
    matrix = xgb.DMatrix([[1.0, -999.0], [2.0, 3.0]], missing=-999.0)
    assert np.isnan(matrix.data[0, 1])
    assert matrix.num_nonmissing() == 3


def test_exact_stump_parity_with_upstream():
    data = np.arange(8, dtype=np.float64).reshape(-1, 1)
    label = np.r_[np.zeros(4), np.ones(4)]
    params = dict(
        n_estimators=1,
        max_depth=1,
        max_bin=8,
        learning_rate=1.0,
        min_child_weight=1.0,
        reg_lambda=0.0,
        tree_method="hist",
        base_score=0.5,
        n_jobs=1,
    )
    ours = xgb.XGBRegressor(**params).fit(data, label, verbose=False).predict(data)
    theirs = upstream.XGBRegressor(**params).fit(data, label).predict(data)
    assert np.array_equal(ours, label)
    assert np.array_equal(theirs, label.astype(np.float32))


@pytest.fixture(scope="module")
def regression_data():
    x, y = make_regression(
        n_samples=1200,
        n_features=9,
        n_informative=7,
        noise=8.0,
        random_state=12,
    )
    return train_test_split(x, y, test_size=0.25, random_state=9)


@pytest.fixture(scope="module")
def classification_data():
    x, y = make_classification(
        n_samples=1400,
        n_features=10,
        n_informative=7,
        n_redundant=1,
        class_sep=1.25,
        flip_y=0.025,
        random_state=14,
    )
    return train_test_split(x, y, test_size=0.25, random_state=9, stratify=y)


def test_regression_parity_on_held_out_data(regression_data):
    x_train, x_test, y_train, y_test = regression_data
    params = dict(
        n_estimators=50, max_depth=4, max_bin=64, learning_rate=0.1,
        tree_method="hist", base_score=0.5, random_state=3, n_jobs=1,
    )
    ours = xgb.XGBRegressor(**params).fit(x_train, y_train, verbose=False)
    theirs = upstream.XGBRegressor(**params).fit(x_train, y_train)
    ours_pred = ours.predict(x_test)
    their_pred = theirs.predict(x_test)
    ours_rmse = mean_squared_error(y_test, ours_pred) ** 0.5
    their_rmse = mean_squared_error(y_test, their_pred) ** 0.5
    assert ours_rmse <= their_rmse * 1.2
    assert np.corrcoef(ours_pred, their_pred)[0, 1] > 0.97


def test_binary_logistic_parity_on_held_out_data(classification_data):
    x_train, x_test, y_train, y_test = classification_data
    params = dict(
        n_estimators=45, max_depth=4, max_bin=64, learning_rate=0.1,
        tree_method="hist", base_score=0.5, random_state=5, n_jobs=1,
    )
    ours = xgb.XGBClassifier(**params).fit(x_train, y_train, verbose=False)
    theirs = upstream.XGBClassifier(**params).fit(x_train, y_train)
    ours_probability = ours.predict_proba(x_test)[:, 1]
    their_probability = theirs.predict_proba(x_test)[:, 1]
    assert abs(ours.score(x_test, y_test) - theirs.score(x_test, y_test)) < 0.04
    assert log_loss(y_test, ours_probability) <= log_loss(y_test, their_probability) * 1.2
    assert np.corrcoef(ours_probability, their_probability)[0, 1] > 0.97
    assert np.allclose(ours.predict_proba(x_test).sum(axis=1), 1.0)


def test_missing_value_behavior_matches_upstream():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(800, 5))
    missing_rows = rng.random(800) < 0.25
    x[missing_rows, 0] = np.nan
    y = np.where(missing_rows, 4.0, 2.0 * x[:, 1] - x[:, 2])
    train_index = np.arange(600)
    test_index = np.arange(600, 800)
    params = dict(
        n_estimators=30, max_depth=3, max_bin=32, learning_rate=0.15,
        tree_method="hist", base_score=0.5, n_jobs=1,
    )
    ours = xgb.XGBRegressor(**params).fit(x[train_index], y[train_index], verbose=False)
    theirs = upstream.XGBRegressor(**params).fit(x[train_index], y[train_index])
    ours_pred = ours.predict(x[test_index])
    their_pred = theirs.predict(x[test_index])
    assert np.corrcoef(ours_pred, their_pred)[0, 1] > 0.97
    assert mean_squared_error(y[test_index], ours_pred) <= (
        mean_squared_error(y[test_index], their_pred) * 1.25
    )


def test_dmatrix_train_api_and_iteration_range(regression_data):
    x_train, x_test, y_train, _ = regression_data
    matrix = xgb.DMatrix(x_train, y_train)
    model = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 3, "max_bin": 32, "eta": 0.1},
        matrix,
        num_boost_round=6,
        verbose_eval=False,
    )
    full = model.predict(xgb.DMatrix(x_test))
    first = model.predict(xgb.DMatrix(x_test), iteration_range=(0, 3))
    second = model.predict(xgb.DMatrix(x_test), iteration_range=(3, 6))
    assert model.num_boosted_rounds() == 6
    assert np.allclose(full, first + second - model.base_margin)


def test_custom_squared_error_objective_matches_builtin(regression_data):
    x_train, _, y_train, _ = regression_data
    matrix = xgb.DMatrix(x_train[:300], y_train[:300])
    params = {"objective": "reg:squarederror", "max_depth": 2, "max_bin": 32, "eta": 0.2}
    builtin = xgb.train(params, matrix, 4, verbose_eval=False)

    def squared_error(margin, dtrain):
        return margin - dtrain.get_label(), np.ones_like(margin)

    custom = xgb.train(params, matrix, 4, obj=squared_error, verbose_eval=False)
    assert np.allclose(builtin.predict(matrix), custom.predict(matrix))


def test_eval_history_matches_upstream_shape(regression_data):
    x_train, x_test, y_train, y_test = regression_data
    ours_history = {}
    ours = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 2, "eval_metric": ["rmse", "mae"]},
        xgb.DMatrix(x_train, y_train),
        7,
        evals=[(xgb.DMatrix(x_test, y_test), "validation")],
        evals_result=ours_history,
        verbose_eval=False,
    )
    upstream_history = {}
    upstream.train(
        {"objective": "reg:squarederror", "max_depth": 2, "tree_method": "hist"},
        upstream.DMatrix(x_train, label=y_train),
        7,
        evals=[(upstream.DMatrix(x_test, label=y_test), "validation")],
        evals_result=upstream_history,
        verbose_eval=False,
    )
    assert list(ours_history["validation"]) == ["rmse", "mae"]
    assert len(ours_history["validation"]["rmse"]) == len(
        upstream_history["validation"]["rmse"]
    )
    assert ours.num_boosted_rounds() == 7


def test_early_stopping_records_best_iteration(regression_data):
    x_train, x_test, y_train, y_test = regression_data
    model = xgb.train(
        {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.15},
        xgb.DMatrix(x_train, y_train),
        25,
        evals=[(xgb.DMatrix(x_test, y_test), "validation")],
        early_stopping_rounds=4,
        verbose_eval=False,
    )
    assert model.best_iteration is not None
    assert model.best_score is not None
    assert 0 <= model.best_iteration < model.num_boosted_rounds()


def test_model_round_trip_and_dump(tmp_path, regression_data):
    x_train, x_test, y_train, _ = regression_data
    model = xgb.XGBRegressor(n_estimators=5, max_depth=2, max_bin=16).fit(
        x_train, y_train, verbose=False
    ).get_booster()
    path = tmp_path / "model.json"
    model.save_model(path)
    restored = xgb.Booster(model_file=path)
    assert np.array_equal(
        model.predict(xgb.DMatrix(x_test)), restored.predict(xgb.DMatrix(x_test))
    )
    parsed = json.loads(model.get_dump(dump_format="json", with_stats=True)[0])
    assert parsed["nodeid"] == 0
    assert "cover" in parsed
    assert "leaf=" in model.get_dump()[0]


def test_model_loader_rejects_unsafe_tree_arrays(tmp_path, regression_data):
    x_train, _, y_train, _ = regression_data
    model = xgb.XGBRegressor(n_estimators=1, max_depth=1).fit(
        x_train, y_train, verbose=False
    ).get_booster()
    path = tmp_path / "model.json"
    model.save_model(path)
    state = json.loads(path.read_text())
    state["features"][0][0] = len(state["feature_names"])
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="out of bounds"):
        xgb.Booster(model_file=path)


def test_leaf_indices_and_importance(classification_data):
    x_train, x_test, y_train, _ = classification_data
    model = xgb.XGBClassifier(n_estimators=8, max_depth=3, max_bin=32).fit(
        x_train, y_train, verbose=False
    )
    leaves = model.apply(x_test[:11])
    assert leaves.shape == (11, 8)
    assert np.all((leaves >= 0) & (leaves < 15))
    assert model.feature_importances_.shape == (x_train.shape[1],)
    assert model.feature_importances_.sum() == pytest.approx(1.0)


def test_sample_weight_changes_fit_in_same_direction_as_upstream():
    x = np.arange(20, dtype=np.float64).reshape(-1, 1)
    y = np.zeros(20)
    y[-1] = 10.0
    weights = np.ones(20)
    weights[-1] = 100.0
    params = dict(n_estimators=3, max_depth=1, learning_rate=0.3, base_score=0.5)
    ours_plain = xgb.XGBRegressor(**params).fit(x, y, verbose=False).predict(x)[-1]
    ours_weighted = xgb.XGBRegressor(**params).fit(
        x, y, sample_weight=weights, verbose=False
    ).predict(x)[-1]
    their_plain = upstream.XGBRegressor(**params).fit(x, y).predict(x)[-1]
    their_weighted = upstream.XGBRegressor(**params).fit(
        x, y, sample_weight=weights
    ).predict(x)[-1]
    assert ours_weighted > ours_plain
    assert their_weighted > their_plain


def test_subsample_and_colsample_are_seed_deterministic(regression_data):
    x_train, x_test, y_train, _ = regression_data
    params = dict(
        n_estimators=8, max_depth=3, max_bin=32, subsample=0.7,
        colsample_bytree=0.6, random_state=42,
    )
    first = xgb.XGBRegressor(**params).fit(x_train, y_train, verbose=False)
    second = xgb.XGBRegressor(**params).fit(x_train, y_train, verbose=False)
    assert np.array_equal(first.predict(x_test), second.predict(x_test))


def test_base_margin_and_output_margin():
    model = xgb.Booster({"objective": "binary:logistic", "base_score": 0.25})
    matrix = xgb.DMatrix(np.zeros((3, 2)))
    assert np.allclose(model.predict(matrix), 0.25)
    margins = np.array([-2.0, 0.0, 2.0])
    with_margin = xgb.DMatrix(np.zeros((3, 2)), base_margin=margins)
    assert np.array_equal(model.predict(with_margin, output_margin=True), margins)


def test_sklearn_parameter_protocol():
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2, custom_flag="kept")
    params = model.get_params()
    assert params["n_estimators"] == 4
    assert params["custom_flag"] == "kept"
    assert model.set_params(max_depth=3, another_flag=2) is model
    assert model.max_depth == 3
    assert model.get_params()["another_flag"] == 2


@pytest.mark.parametrize(
    "params,message",
    [
        ({"tree_method": "exact"}, "tree_method"),
        ({"booster": "dart"}, "booster"),
        ({"monotone_constraints": "(1,0)"}, "monotone"),
    ],
)
def test_unsupported_training_modes_are_explicit(params, message):
    matrix = xgb.DMatrix([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(NotImplementedError, match=message):
        xgb.train(params, matrix, 1, verbose_eval=False)


def test_classifier_rejects_nonbinary_labels():
    with pytest.raises(ValueError, match="labels 0 and 1"):
        xgb.XGBClassifier(n_estimators=1).fit(
            [[0.0], [1.0], [2.0]], [0, 1, 2], verbose=False
        )
