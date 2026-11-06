from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])
chosen_from = config["chosen_from"]
max_depth = None if config["max_depth"] == "unlimited" else int(config["max_depth"])

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp,
)

model = Pipeline([
    ("selector", SelectKBest(score_func=f_classif, k=selector_k)),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        ),
    ),
])
model.fit(X_train, y_train)

models_folder = Path("models")
models_folder.mkdir(exist_ok=True)

joblib.dump(model, models_folder / "delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv(
    models_folder / "delay_threshold.csv",
    index=False,
)
pd.DataFrame({"feature": FEATURES}).to_csv(
    models_folder / "feature_order.csv",
    index=False,
)

raw_df = pd.read_csv("data/raw_shipments.csv")
_, medians = clean_shipments(raw_df)
impute_values = pd.DataFrame([
    {"column": column, "median": medians[column]}
    for column in [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]
])
impute_values.to_csv(models_folder / "impute_values.csv", index=False)

reloaded_model = joblib.load(models_folder / "delay_model.pkl")
memory_probabilities = model.predict_proba(X_test)[:, 1]
reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]
max_probability_difference = np.max(
    np.abs(memory_probabilities - reloaded_probabilities)
)

reload_check = pd.DataFrame([
    {"metric": "test_rows_scored", "value": len(X_test)},
    {
        "metric": "reload_max_abs_probability_diff",
        "value": max_probability_difference,
    },
    {"metric": "threshold", "value": threshold},
])
reload_check.to_csv("outputs/part_d_reload_check.csv", index=False)

print("Packaged configuration:", chosen_from)
print("Test rows checked:", len(X_test))
print("Maximum probability difference:", max_probability_difference)
