import pandas as pd
from src.shipment_prep import FEATURES, clean_shipments, build_features

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

before_df = raw.drop_duplicates().copy()
check_metrics = {
    "raw_rows": raw.shape[0],
    "duplicate_rows_removed": raw.shape[0] - before_df.shape[0],
    "rows_after_dedup": before_df.shape[0],
    "missing_distance_km_before": before_df["distance_km"].isna().sum(),
    "missing_parcel_weight_before": before_df["parcel_weight_kg"].isna().sum(),
    "sentinel_weight_rows": (before_df["parcel_weight_kg"] > 100).sum(),
    "missing_declared_value_before": before_df["declared_value"].isna().sum(),
    "missing_payment_mode_before": before_df["payment_mode"].isna().sum(),
    "service_type_spellings_before": before_df["service_type"].nunique(),
}

clean_df, medians = clean_shipments(raw)
results_df = build_features(clean_df, hubs)

results_df["is_delayed"] = (
    results_df["delivery_hours"] > results_df["promised_hours"]
).astype(int)

check_metrics["service_type_spellings_after"] = clean_df["service_type"].nunique()
check_metrics["rows_after_hub_merge"] = results_df.shape[0]
check_metrics["delayed_count"] = results_df["is_delayed"].sum()
check_metrics["delay_rate"] = results_df["is_delayed"].mean()
check_metrics["median_distance_km"] = medians["distance_km"]
check_metrics["median_parcel_weight_kg"] = medians["parcel_weight_kg"]
check_metrics["median_declared_value_inr"] = medians["declared_value_inr"]
check_metrics["declared_value_mean_inr"] = results_df["declared_value_inr"].mean()
check_metrics["declared_value_median_inr"] = results_df["declared_value_inr"].median()
check_metrics["log_declared_value_mean"] = results_df["log_declared_value"].mean()
check_metrics["log_declared_value_median"] = results_df["log_declared_value"].median()

columns_to_check = FEATURES + ["delivery_hours", "is_delayed"]
check_metrics["nulls_in_features_and_target"] = (
    results_df[columns_to_check].isna().sum().sum()
)

hours_q1 = results_df["delivery_hours"].quantile(0.25)
hours_q3 = results_df["delivery_hours"].quantile(0.75)
hours_iqr = hours_q3 - hours_q1
hours_lower_bound = hours_q1 - 1.5 * hours_iqr
hours_upper_bound = hours_q3 + 1.5 * hours_iqr
hours_outlier_count = (
    (results_df["delivery_hours"] < hours_lower_bound)
    | (results_df["delivery_hours"] > hours_upper_bound)
).sum()

check_metrics["hours_q1"] = hours_q1
check_metrics["hours_q3"] = hours_q3
check_metrics["hours_lower_bound"] = hours_lower_bound
check_metrics["hours_upper_bound"] = hours_upper_bound
check_metrics["hours_outlier_count"] = hours_outlier_count

results_df.to_csv("data/modelling_table.csv", index=False)

check_table = pd.DataFrame(
    list(check_metrics.items()),
    columns=["metric", "value"],
)

rounding_digits = {
    "delay_rate": 4,
    "declared_value_mean_inr": 2,
    "declared_value_median_inr": 2,
    "log_declared_value_mean": 3,
    "log_declared_value_median": 3,
    "hours_q1": 2,
    "hours_q3": 2,
    "hours_lower_bound": 2,
    "hours_upper_bound": 2,
}

for metric, digits in rounding_digits.items():
    check_table.loc[check_table["metric"] == metric, "value"] = check_table.loc[
        check_table["metric"] == metric, "value"
    ].round(digits)

check_table.to_csv("outputs/part_a_checks.csv", index=False)


def summarize_group(data, group_column, dimension_name):
    summary = data.groupby(group_column, as_index=False).agg(
        shipments=("shipment_id", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    )

    summary["dimension"] = dimension_name
    summary = summary.rename(columns={group_column: "group"})

    return summary[
        ["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]
    ]


service_summary = summarize_group(results_df, "service_type", "service_type")
tier_summary = summarize_group(results_df, "hub_tier", "hub_tier")

distance_bins = [0, 200, 400, 600, 2000]
distance_labels = ["0-200", "201-400", "401-600", "601+"]

results_df["distance_band"] = pd.cut(
    results_df["distance_km"],
    bins=distance_bins,
    labels=distance_labels,
    right=True,
    include_lowest=True,
)

distance_summary = summarize_group(results_df, "distance_band", "distance_band")

delay_by_group = pd.concat(
    [service_summary, tier_summary, distance_summary],
    ignore_index=True,
)

delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)

delay_by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)

numeric_data = results_df.select_dtypes(include="number")
correlation_matrix = numeric_data.corr()

correlation_matrix.to_csv("outputs/part_a_correlations.csv")
