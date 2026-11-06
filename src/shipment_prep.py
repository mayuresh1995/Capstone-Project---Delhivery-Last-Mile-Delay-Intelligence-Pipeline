import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):

    clean_df = raw
    clean_df = clean_df.drop_duplicates()
    clean_df['service_type'] = clean_df['service_type'].str.strip().str.title()
    clean_df['payment_mode'] = clean_df['payment_mode'].fillna('Unknown')
    clean_df['declared_value_inr'] = clean_df['declared_value'].replace("₹", "", regex=True).replace(",", "", regex=True).astype(float)
    clean_df = clean_df.drop(columns='declared_value')
    clean_df['dispatch_time'] = pd.to_datetime(clean_df['dispatch_time'])
    clean_df.loc[clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    if medians is None:
        medians = {
            "distance_km": round(float(clean_df["distance_km"].median()),2),
            "parcel_weight_kg": round(float(clean_df["parcel_weight_kg"].median()),2),
            "declared_value_inr": round(float(clean_df["declared_value_inr"].median()),2)
            }
    for column, median_value in medians.items():
        clean_df[column] = clean_df[column].fillna(median_value)
  
    return (clean_df, medians)

def build_features(clean, hubs):

    result_df = clean.merge(hubs[['hub_id','hub_city', 'region', 'hub_tier', 'hub_daily_capacity']], on='hub_id', how='left')
    result_df['promised_hours'] = result_df['service_type'].map(PROMISED_HOURS).astype(int)
    result_df['is_cod'] = (result_df['payment_mode'] == "COD").astype(int)
    result_df["is_weekend"] = (result_df['dispatch_time'].dt.day_of_week >= 5).astype(int)
    result_df['dispatch_hour'] = (result_df['dispatch_time'].dt.hour).astype(int)
    result_df["log_declared_value"] = np.log(result_df["declared_value_inr"])

    return result_df