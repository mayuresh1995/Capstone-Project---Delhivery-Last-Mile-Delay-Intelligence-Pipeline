import pandas as pd
import numpy as np
from src.shipment_prep import FEATURES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["delivery_hours"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

metrics = {
    "train_rows": X_train.shape[0],
    "val_rows": X_val.shape[0],
    "test_rows": X_test.shape[0],
}

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURES)
test_scaled_df = pd.DataFrame(X_test_scaled, columns=FEATURES)

metrics["train_scaled_max_abs_mean"] = train_scaled_df.mean().abs().max()
metrics["test_scaled_max_abs_mean"] = test_scaled_df.mean().abs().max()

baseline_prediction = np.full(len(y_val), y_train.mean())

baseline_metrics = {
    "mae": mean_absolute_error(y_val, baseline_prediction),
    "mse": mean_squared_error(y_val, baseline_prediction),
    "rmse": np.sqrt(mean_squared_error(y_val, baseline_prediction)),
    "r2": r2_score(y_val, baseline_prediction),
}

for metric_name, metric_value in baseline_metrics.items():
    metrics[f"baseline_val_{metric_name}"] = metric_value

checks_df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
checks_df.to_csv("outputs/part_b_checks.csv", index=False)

max_depth = [1, 2, 3, 4, 6, 8, 12, None]
tree_results = []

for d in max_depth:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)

    train_pred = tree.predict(X_train)
    val_pred = tree.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    tree_results.append({"depth": d, "train_rmse": train_rmse, "val_rmse": val_rmse})

tree_table = pd.DataFrame(
    [
        {
            "max_depth": "unlimited" if result["depth"] is None else result["depth"],
            "train_rmse": result["train_rmse"],
            "val_rmse": result["val_rmse"],
        }
        for result in tree_results
    ]
)

tree_table.to_csv("outputs/part_b_tree_depth.csv", index=False)

best_tree = min(
    tree_results,
    key=lambda result: (
        result["val_rmse"],
        result["depth"] if result["depth"] is not None else float("inf"),
    ),
)

best_tree_depth = best_tree["depth"]
print(
    "Best tree depth:", best_tree_depth if best_tree_depth is not None else "unlimited"
)

pipeline = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
pipeline.fit(X_train, y_train)
y_pred_lr = pipeline.predict(X_val)

dtr = DecisionTreeRegressor(max_depth=best_tree_depth, random_state=42)
dtr.fit(X_train, y_train)
y_pred_dtr = dtr.predict(X_val)

rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_val)

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_val)

model_predictions = {
    "mean_baseline": baseline_prediction,
    "linear_regression": y_pred_lr,
    "decision_tree": y_pred_dtr,
    "random_forest": y_pred_rfr,
    "gradient_boosting": y_pred_gbr,
}

metric_rows = []

for model_name, predictions in model_predictions.items():
    mse = mean_squared_error(y_val, predictions)

    metric_rows.append(
        {
            "model": model_name,
            "split": "val",
            "MAE": mean_absolute_error(y_val, predictions),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(y_val, predictions),
        }
    )

metrics_df = pd.DataFrame(
    metric_rows,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
)

learning_rates = [0.01, 0.1, 1.0]
learning_rate_results = []

for lr in learning_rates:
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)

    learning_rate_results.append(
        {
            "learning_rate": lr,
            "val_rmse": np.sqrt(mean_squared_error(y_val, val_prediction)),
            "val_r2": r2_score(y_val, val_prediction),
        }
    )

learning_rate_table = pd.DataFrame(
    learning_rate_results,
    columns=["learning_rate", "val_rmse", "val_r2"],
)

learning_rate_table.to_csv(
    "outputs/part_b_gb_learning_rate.csv",
    index=False,
)

candidate_metrics = metrics_df[metrics_df["model"] != "mean_baseline"]
candidate_metrics = candidate_metrics.sort_values("RMSE")

top_two_names = candidate_metrics["model"].head(2).tolist()
final_model_name = candidate_metrics.iloc[0]["model"]

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    "linear_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
    ),
}

cv_rows = []

for model_name in top_two_names:
    scores = cross_val_score(
        cv_models[model_name],
        X_train_val,
        y_train_val,
        cv=cv,
    )

    cv_rows.append(
        {
            "model": model_name,
            "mean_r2": scores.mean(),
            "std_r2": scores.std(),
        }
    )

cv_table = pd.DataFrame(
    cv_rows,
    columns=["model", "mean_r2", "std_r2"],
)
cv_table.to_csv("outputs/part_b_cv.csv", index=False)

fitted_models = {
    "linear_regression": pipeline,
    "decision_tree": dtr,
    "random_forest": rfr,
    "gradient_boosting": gbr,
}

final_prediction = fitted_models[final_model_name].predict(X_test)
test_mse = mean_squared_error(y_test, final_prediction)

test_row = pd.DataFrame(
    [
        {
            "model": final_model_name,
            "split": "test",
            "MAE": mean_absolute_error(y_test, final_prediction),
            "MSE": test_mse,
            "RMSE": np.sqrt(test_mse),
            "R2": r2_score(y_test, final_prediction),
        }
    ]
)

metrics_df = pd.concat([metrics_df, test_row], ignore_index=True)
metrics_df.to_csv("outputs/part_b_metrics.csv", index=False)
