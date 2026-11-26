
"""
Build cleaned and feature-engineered datasets from the raw daily
production data.

Input:

    data/raw/production_data.csv

Outputs:

    data/processed/production_data_clean.csv
    data/processed/production_features.csv
    data/processed/summary.json

Run from the project root:

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe src\\build_processed_data.py
"""

import json

import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
CLEAN_PATH = OUTPUT_DIR / "production_data_clean.csv"
FEATURES_PATH = OUTPUT_DIR / "production_features.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

PRODUCTS = ("A", "B", "C")

REQUIRED_COLUMNS = [
    "day",
    "demand_A", "demand_B", "demand_C",
    "raw_material_capacity",
    "machine_hours_capacity",
    "production_budget",
    "storage_capacity",
    "price_A", "cost_A", "material_A", "hours_A",
    "price_B", "cost_B", "material_B", "hours_B",
    "price_C", "cost_C", "material_C", "hours_C",
]


def load_raw():
    """Read the raw dataset and check its schema."""

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_PATH}")

    frame = pd.read_csv(RAW_PATH)
    frame.columns = [str(column).strip() for column in frame.columns]

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Raw dataset is missing required columns: "
            + ", ".join(missing)
        )

    return frame[REQUIRED_COLUMNS].copy()


def clean(frame):
    """Validate, type, deduplicate, and sort the raw rows."""

    frame = frame.copy()

    frame.columns = [str(column).strip() for column in frame.columns]

    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.isna().any().any():
        bad = frame.isna().sum()
        bad = bad[bad > 0]
        detail = ", ".join(
            f"{column}: {int(count)}"
            for column, count in bad.items()
        )
        raise ValueError(f"Non-numeric or missing values found: {detail}")

    frame = frame.drop_duplicates(subset="day", keep="last")
    frame = frame.sort_values("day").reset_index(drop=True)

    if frame["day"].duplicated().any():
        raise ValueError("Duplicate day values remain after cleaning")

    if (frame[REQUIRED_COLUMNS] < 0).any().any():
        raise ValueError("Negative values are not allowed")

    return frame


def add_features(frame):
    """Derive demand, resource, cost, and capacity features."""

    data = frame.copy()

    data["total_demand"] = data[
        [f"demand_{product}" for product in PRODUCTS]
    ].sum(axis=1)

    for product in PRODUCTS:
        data[f"demand_share_{product}"] = (
            data[f"demand_{product}"] / data["total_demand"]
        )

    data["total_revenue"] = sum(
        data[f"demand_{product}"] * data[f"price_{product}"]
        for product in PRODUCTS
    )

    data["total_production_cost"] = sum(
        data[f"demand_{product}"] * data[f"cost_{product}"]
        for product in PRODUCTS
    )

    data["gross_margin"] = (
        data["total_revenue"] - data["total_production_cost"]
    )

    data["gross_margin_pct"] = (
        data["gross_margin"] / data["total_revenue"]
    )

    data["total_material_required"] = sum(
        data[f"demand_{product}"] * data[f"material_{product}"]
        for product in PRODUCTS
    )

    data["total_machine_hours_required"] = sum(
        data[f"demand_{product}"] * data[f"hours_{product}"]
        for product in PRODUCTS
    )

    data["raw_material_utilization"] = (
        data["total_material_required"] / data["raw_material_capacity"]
    )

    data["machine_hours_utilization"] = (
        data["total_machine_hours_required"]
        / data["machine_hours_capacity"]
    )

    data["budget_utilization"] = (
        data["total_production_cost"] / data["production_budget"]
    )

    data["storage_utilization"] = (
        data["total_demand"] / data["storage_capacity"]
    )

    data["material_headroom"] = (
        data["raw_material_capacity"] - data["total_material_required"]
    )

    data["machine_hours_headroom"] = (
        data["machine_hours_capacity"]
        - data["total_machine_hours_required"]
    )

    data["budget_headroom"] = (
        data["production_budget"] - data["total_production_cost"]
    )

    data["capacity_feasible"] = (
        (data["raw_material_utilization"] <= 1.0)
        & (data["machine_hours_utilization"] <= 1.0)
        & (data["budget_utilization"] <= 1.0)
    )

    data["demand_7day_avg"] = (
        data["total_demand"].rolling(window=7, min_periods=1).mean()
    )

    return data


def build_summary(clean_frame, feature_frame, duplicate_days_removed):
    """Describe the processed outputs."""

    return {
        "source": str(RAW_PATH.relative_to(BASE_DIR)).replace("\\", "/"),
        "rows": int(len(clean_frame)),
        "raw_columns": int(len(REQUIRED_COLUMNS)),
        "feature_columns": int(len(feature_frame.columns)),
        "day_start": int(clean_frame["day"].min()),
        "day_end": int(clean_frame["day"].max()),
        "duplicate_days_removed": int(duplicate_days_removed),
        "missing_values": 0,
        "negative_values": 0,
        "capacity_feasible_days": int(
            feature_frame["capacity_feasible"].sum()
        ),
        "capacity_infeasible_days": int(
            (~feature_frame["capacity_feasible"]).sum()
        ),
        "outputs": {
            "clean": str(CLEAN_PATH.relative_to(BASE_DIR)).replace("\\", "/"),
            "features": str(
                FEATURES_PATH.relative_to(BASE_DIR)
            ).replace("\\", "/"),
        },
    }


def main():

    raw = load_raw()
    duplicate_days_removed = int(raw["day"].duplicated().sum())
    clean_frame = clean(raw)
    feature_frame = add_features(clean_frame)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_frame.to_csv(CLEAN_PATH, index=False)
    feature_frame.to_csv(FEATURES_PATH, index=False)

    summary = build_summary(
        clean_frame,
        feature_frame,
        duplicate_days_removed
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Raw rows: {len(raw)}")
    print(f"Clean rows: {len(clean_frame)}")
    print(f"Feature columns: {len(feature_frame.columns)}")
    print(f"Wrote {CLEAN_PATH.relative_to(BASE_DIR)}")
    print(f"Wrote {FEATURES_PATH.relative_to(BASE_DIR)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
