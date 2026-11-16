
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT_DIR / "data" / "raw" / "production_data.csv"
)

columns = [
    "demand_A",
    "demand_B",
    "demand_C",
    "raw_material_capacity",
    "machine_hours_capacity",
    "production_budget"
]

print("\n========== ENVIRONMENT DATA ANALYSIS ==========")

print("\nMinimum values:")
print(df[columns].min())

print("\nMaximum values:")
print(df[columns].max())

print("\nAverage values:")
print(df[columns].mean())

print("\n95th percentile:")
print(df[columns].quantile(0.95))

print("\n========== ANALYSIS COMPLETED ==========")
