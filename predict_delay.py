import joblib
import numpy as np
import pandas as pd

from src.shipment_prep import build_features, clean_shipments

new_parcels = pd.read_csv("data/new_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")
model = joblib.load("models/delay_model.pkl")

threshold_table = pd.read_csv("models/delay_threshold.csv")
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute_values = pd.read_csv("models/impute_values.csv")
hub_segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

threshold = float(threshold_table.loc[0, "threshold"])
stored_medians = dict(zip(impute_values["column"], impute_values["median"]))
expected_median_columns = {
    "distance_km",
    "parcel_weight_kg",
    "declared_value_inr",
}
if set(stored_medians) != expected_median_columns:
    raise ValueError("Stored imputation medians do not contain the expected columns.")

new_parcels["_file_order"] = np.arange(len(new_parcels))
clean_parcels, _ = clean_shipments(new_parcels, medians=stored_medians)
parcels_with_features = build_features(clean_parcels, hubs)
parcels_with_features = parcels_with_features.sort_values(
    "_file_order",
    kind="stable",
)

missing_features = set(feature_order) - set(parcels_with_features.columns)
if missing_features:
    raise ValueError(f"Missing model features: {sorted(missing_features)}")

model_features = parcels_with_features[feature_order]
if model_features.isna().any().any():
    raise ValueError("Scoring features contain missing values.")

positive_class_index = model.classes_.tolist().index(1)
raw_probabilities = model.predict_proba(model_features)[:, positive_class_index]
rounded_probabilities = np.round(raw_probabilities, 4)
flagged = (rounded_probabilities >= threshold).astype(int)

scored_parcels = parcels_with_features[["shipment_id", "hub_id", "_file_order"]].copy()
scored_parcels["delay_probability"] = rounded_probabilities
scored_parcels["flagged"] = flagged
scored_parcels = scored_parcels.merge(
    hub_segments,
    on="hub_id",
    how="left",
    validate="many_to_one",
)

if scored_parcels["segment"].isna().any():
    unknown_hubs = scored_parcels.loc[
        scored_parcels["segment"].isna(), "hub_id"
    ].unique()
    raise ValueError(f"No segment found for hubs: {unknown_hubs.tolist()}")

scored_parcels["action"] = np.select(
    [
        (scored_parcels["flagged"] == 1) & (scored_parcels["segment"] == "Escalate"),
        scored_parcels["flagged"] == 1,
    ],
    [
        "Escalate to hub manager",
        "Proactive customer call",
    ],
    default="No action",
)

scored_parcels = scored_parcels.sort_values("_file_order", kind="stable")
scored_parcels = scored_parcels[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action",
    ]
]
scored_parcels.to_csv(
    "outputs/new_shipment_actions.csv",
    index=False,
    float_format="%.4f",
)

print("Scored parcels:", len(scored_parcels))
print("Flagged parcels:", int(scored_parcels["flagged"].sum()))
