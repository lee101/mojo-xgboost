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

print(model.predict(X[:5]))

dtrain = xgb.DMatrix(X, label=y)
booster = xgb.train(
    {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": 4},
    dtrain,
    num_boost_round=20,
    verbose_eval=False,
)
print(booster.predict(xgb.DMatrix(X[:5])))
