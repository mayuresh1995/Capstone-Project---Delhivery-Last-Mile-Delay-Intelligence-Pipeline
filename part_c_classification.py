import pandas as pd
from src.shipment_prep import FEATURES, NOISE_COLUMNS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

df = pd.read_csv("data/modelling_table.csv")

X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


def get_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total

    precision_denominator = tp + fp
    precision = tp / precision_denominator if precision_denominator else 0

    recall_denominator = tp + fn
    recall = tp / recall_denominator if recall_denominator else 0

    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


model_rows = []

baseline_predictions = [0] * len(y_val)
model_rows.append(
    {
        "model": "blind_all_on_time",
        **get_metrics(y_val, baseline_predictions),
    }
)

logistic_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression()),
    ]
)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_val)

model_rows.append(
    {
        "model": "logistic_regression",
        **get_metrics(y_val, logistic_predictions),
    }
)

best_depth = None
best_f1 = -1

for depth in [2, 3, 4, 5, 6, 8, 10]:
    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42,
    )
    tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_val)
    tree_metrics = get_metrics(y_val, tree_predictions)

    if tree_metrics["f1"] > best_f1:
        best_f1 = tree_metrics["f1"]
        best_depth = depth
        best_tree_predictions = tree_predictions

model_rows.append(
    {
        "model": f"decision_tree_depth_{best_depth}",
        **get_metrics(y_val, best_tree_predictions),
    }
)

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_val)

model_rows.append(
    {
        "model": "random_forest_default",
        **get_metrics(y_val, forest_predictions),
    }
)

models_table = pd.DataFrame(
    model_rows,
    columns=["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall", "f1"],
)
models_table.to_csv("outputs/part_c_models.csv", index=False)

all_features_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
all_features_forest.fit(X_train, y_train)

feature_selector = SelectKBest(score_func=f_classif, k="all")
feature_selector.fit(X_train, y_train)

feature_scores = pd.DataFrame(
    {
        "feature": FEATURES,
        "f_score": feature_selector.scores_,
        "rf_importance": all_features_forest.feature_importances_,
    }
)

feature_scores["f_rank"] = (
    feature_scores["f_score"]
    .rank(
        ascending=False,
        method="min",
    )
    .astype(int)
)

feature_scores["is_noise"] = feature_scores["feature"].isin(NOISE_COLUMNS)

feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]
feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

selectk_rows = []

for k in [5, 7, 10, 12, 17]:
    model = Pipeline(
        [
            ("select", SelectKBest(score_func=f_classif, k=k)),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    selectk_rows.append(
        {
            "k": k,
            "val_f1": f1_score(y_val, predictions, zero_division=0),
        }
    )

selectk_results = pd.DataFrame(
    selectk_rows,
    columns=["k", "val_f1"],
)
selectk_results.to_csv("outputs/part_c_selectk.csv", index=False)

selector = SelectKBest(score_func=f_classif, k=7)
X_train_7 = selector.fit_transform(X_train, y_train)
X_val_7 = selector.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
)
grid_search.fit(X_train_7, y_train)
grid_predictions = grid_search.best_estimator_.predict(X_val_7)

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=9,
    cv=5,
    random_state=42,
)
random_search.fit(X_train_7, y_train)
random_predictions = random_search.best_estimator_.predict(X_val_7)

default_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
)
default_forest.fit(X_train_7, y_train)
default_predictions = default_forest.predict(X_val_7)

tuning_rows = [
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 27 * 5,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": f1_score(y_val, grid_predictions, zero_division=0),
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 9 * 5,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": f1_score(y_val, random_predictions, zero_division=0),
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "n_estimators=100, max_depth=None, min_samples_leaf=1",
        "best_cv_accuracy": None,
        "val_f1": f1_score(y_val, default_predictions, zero_division=0),
    },
]

tuning_table = pd.DataFrame(
    tuning_rows,
    columns=[
        "search",
        "candidates",
        "cv_folds",
        "total_fits",
        "best_params",
        "best_cv_accuracy",
        "val_f1",
    ],
)

tuning_table.to_csv("outputs/part_c_tuning.csv", index=False)

tie_priority = {
    "default_forest_same_7_features": 0,
    "grid_search": 1,
    "randomized_search": 2,
}

tuning_table["tie_priority"] = tuning_table["search"].map(tie_priority)
best_row = tuning_table.sort_values(
    ["val_f1", "tie_priority"],
    ascending=[False, True],
).iloc[0]

chosen_search = best_row["search"]

if chosen_search == "grid_search":
    chosen_params = grid_search.best_params_
elif chosen_search == "randomized_search":
    chosen_params = random_search.best_params_
else:
    chosen_params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
    }

final_model = Pipeline(
    [
        ("selector", SelectKBest(score_func=f_classif, k=7)),
        (
            "model",
            RandomForestClassifier(
                random_state=42,
                **chosen_params,
            ),
        ),
    ]
)

final_model.fit(X_train, y_train)

print("Selected configuration:", chosen_search)
print("Selected parameters:", chosen_params)

validation_probabilities = final_model.predict_proba(X_val)[:, 1]

threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold_rows = []

for threshold in threshold_values:
    predictions = (validation_probabilities >= threshold).astype(int)

    threshold_rows.append(
        {
            "threshold": threshold,
            **get_metrics(y_val, predictions),
        }
    )

threshold_table = pd.DataFrame(
    threshold_rows,
    columns=[
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)

threshold_table.to_csv("outputs/part_c_thresholds.csv", index=False)

eligible_thresholds = threshold_table[threshold_table["recall"] >= 0.80]

if eligible_thresholds.empty:
    alert_threshold = 0.1
else:
    alert_threshold = eligible_thresholds["threshold"].max()

print("Chosen alert threshold:", alert_threshold)

test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= alert_threshold).astype(int)

final_test_row = {
    "threshold": alert_threshold,
    **get_metrics(y_test, test_predictions),
}

final_test_table = pd.DataFrame(
    [final_test_row],
    columns=[
        "threshold",
        "TN",
        "FP",
        "FN",
        "TP",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ],
)
final_test_table.to_csv("outputs/part_c_final_test.csv", index=False)

final_config = pd.DataFrame(
    [
        {
            "selector_k": 7,
            "n_estimators": chosen_params["n_estimators"],
            "max_depth": (
                "unlimited"
                if chosen_params["max_depth"] is None
                else chosen_params["max_depth"]
            ),
            "min_samples_leaf": chosen_params["min_samples_leaf"],
            "threshold": alert_threshold,
            "chosen_from": chosen_search,
        }
    ]
)
final_config.to_csv("outputs/part_c_final_config.csv", index=False)
