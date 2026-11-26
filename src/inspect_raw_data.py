import pandas as pd

df = pd.read_csv("data/raw/production_data.csv")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDemand averages:")
print(df[["demand_A", "demand_B", "demand_C"]].mean())

print("\nResource averages:")
print(
    df[
        [
            "raw_material_capacity",
            "machine_hours_capacity",
            "production_budget"
        ]
    ].mean()
)