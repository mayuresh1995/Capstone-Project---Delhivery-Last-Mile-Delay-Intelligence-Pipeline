import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

df = pd.read_csv("data/modelling_table.csv")

hub_profile = (
    df.groupby("hub_id")
    .agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    )
    .reset_index()
    .sort_values("hub_id")
)

hub_profile.to_csv("outputs/hub_profile.csv", index=False)

profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops",
]

scaler = StandardScaler()
scaled_profiles = scaler.fit_transform(hub_profile[profile_columns])

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)
hub_profile["segment"] = kmeans.fit_predict(scaled_profiles)

separation_rows = []

for cluster_count in [2, 3, 4, 5]:
    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10,
    )
    labels = model.fit_predict(scaled_profiles)

    within_distances = []
    across_distances = []

    for first, second in combinations(range(len(scaled_profiles)), 2):
        distance = np.linalg.norm(scaled_profiles[first] - scaled_profiles[second])

        if labels[first] == labels[second]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    cluster_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    cluster_sizes_text = "-".join(str(size) for size in cluster_sizes)

    separation_rows.append(
        {
            "K": cluster_count,
            "across_over_within": (
                np.mean(across_distances) / np.mean(within_distances)
            ),
            "cluster_sizes": cluster_sizes_text,
        }
    )

separation_table = pd.DataFrame(
    separation_rows,
    columns=["K", "across_over_within", "cluster_sizes"],
)
separation_table.to_csv("outputs/part_d_separation.csv", index=False)

# Give cluster names according to their average hub delay rate.
cluster_delay_rates = hub_profile.groupby("segment")["delay_rate"].mean().sort_values()

segment_names = ["Standard", "Watch", "Escalate"]
segment_name_map = dict(zip(cluster_delay_rates.index, segment_names))

hub_profile["segment"] = hub_profile["segment"].map(segment_name_map)

# Add hub city and tier information.
hubs = pd.read_csv("data/hubs.csv")
hub_segments = hub_profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left",
)

hub_segments = hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]]
hub_segments.to_csv("outputs/hub_segments.csv", index=False)


# Summarize the hub-level values within each named segment.
segment_profile = (
    hub_profile.groupby("segment")
    .agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    )
    .reset_index()
)

segment_profile = segment_profile[
    [
        "segment",
        "hubs",
        "mean_delivery_hours",
        "delay_rate",
        "mean_distance_km",
        "mean_n_stops",
    ]
]
segment_profile.to_csv("outputs/segment_profile.csv", index=False)


# Find hubs whose segment does not match the tier's expected segment.
expected_segment_by_tier = {
    1: "Standard",
    2: "Watch",
    3: "Escalate",
}

hub_segments["expected_segment"] = hub_segments["hub_tier"].map(
    expected_segment_by_tier
)
mismatches = hub_segments[hub_segments["segment"] != hub_segments["expected_segment"]]
mismatches = mismatches[["hub_id", "segment", "hub_city", "hub_tier"]]
mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
