
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================

RANDOM_SEED = 42
TOTAL_DAYS = 3000

np.random.seed(RANDOM_SEED)

# Product information
products = {
    "A": {
        "selling_price": 500,
        "production_cost": 250,
        "raw_material": 3,
        "machine_hours": 2
    },
    "B": {
        "selling_price": 400,
        "production_cost": 180,
        "raw_material": 2,
        "machine_hours": 1.5
    },
    "C": {
        "selling_price": 300,
        "production_cost": 120,
        "raw_material": 1,
        "machine_hours": 1
    }
}

# Average daily demand
average_demand = {
    "A": 25,
    "B": 30,
    "C": 20
}

# ==========================================
# 2. GENERATE DAILY FACTORY DATA
# ==========================================

records = []

for day in range(1, TOTAL_DAYS + 1):

    # Weekly demand variation
    weekly_factor = 1 + 0.10 * np.sin(
        2 * np.pi * day / 7
    )

    # Long-term demand variation
    seasonal_factor = 1 + 0.08 * np.sin(
        2 * np.pi * day / 90
    )

    record = {
        "day": day
    }

    # Generate product demand
    for product, mean in average_demand.items():

        demand = np.random.normal(
            loc=mean * weekly_factor * seasonal_factor,
            scale=mean * 0.20
        )

        # Keep demand positive
        record[f"demand_{product}"] = max(
            0, int(round(demand))
        )

    # ======================================
    # 3. RESOURCE AVAILABILITY
    # ======================================

    record["raw_material_capacity"] = max(
        150,
        int(np.random.normal(180, 10))
    )

    record["machine_hours_capacity"] = max(
        85,
        round(np.random.normal(100, 5), 2)
    )

    record["production_budget"] = max(
        17000,
        round(np.random.normal(20000, 1000), 2)
    )

    record["storage_capacity"] = 200

    # ======================================
    # 4. PRODUCT ECONOMICS
    # ======================================

    for product, details in products.items():

        record[f"price_{product}"] = details["selling_price"]

        record[f"cost_{product}"] = details["production_cost"]

        record[f"material_{product}"] = details["raw_material"]

        record[f"hours_{product}"] = details["machine_hours"]

    records.append(record)


# ==========================================
# 5. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(records)

# ==========================================
# 6. SAVE DATASET
# ==========================================

output_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "production_data.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

# ==========================================
# 7. DATASET INFORMATION
# ==========================================

print("\n========== DATASET GENERATED ==========")

print("Dataset shape:", df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDemand statistics:")
print(
    df[["demand_A", "demand_B", "demand_C"]].describe()
)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nSaved dataset to:")
print(output_path)

print("\n=======================================")
