
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. LOAD DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "raw" / "production_data.csv"

df = pd.read_csv(data_path)

# Folder for graphs
output_dir = BASE_DIR / "reports" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

print("\n========== EDA REPORT ==========")

# ==========================================
# 2. DEMAND ANALYSIS
# ==========================================

demand_columns = [
    "demand_A",
    "demand_B",
    "demand_C"
]

print("\nDemand Statistics:")
print(df[demand_columns].describe())

# ==========================================
# 3. DEMAND DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

for column in demand_columns:
    plt.hist(
        df[column],
        bins=20,
        alpha=0.5,
        label=column
    )

plt.title("Product Demand Distribution")
plt.xlabel("Daily Demand")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(output_dir / "demand_distribution.png")
plt.show()

# ==========================================
# 4. DAILY DEMAND TREND
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["day"],
    df["demand_A"].rolling(30).mean(),
    label="Product A"
)

plt.plot(
    df["day"],
    df["demand_B"].rolling(30).mean(),
    label="Product B"
)

plt.plot(
    df["day"],
    df["demand_C"].rolling(30).mean(),
    label="Product C"
)

plt.title("30-Day Rolling Average Demand")
plt.xlabel("Day")
plt.ylabel("Average Demand")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(output_dir / "demand_trend.png")
plt.show()

# ==========================================
# 5. RESOURCE ANALYSIS
# ==========================================

resource_columns = [
    "raw_material_capacity",
    "machine_hours_capacity",
    "production_budget"
]

print("\nResource Statistics:")
print(df[resource_columns].describe())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, column in zip(axes, resource_columns):
    ax.hist(df[column], bins=25)
    ax.set_title(column.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(output_dir / "resource_distribution.png")
plt.show()

# ==========================================
# 6. PRODUCT PROFITABILITY
# ==========================================

print("\nProduct Economics:")

for product in ["A", "B", "C"]:

    price = df[f"price_{product}"].iloc[0]
    cost = df[f"cost_{product}"].iloc[0]

    profit = price - cost

    margin = (profit / price) * 100

    print(f"\nProduct {product}")
    print("Selling Price:", price)
    print("Production Cost:", cost)
    print("Gross Profit per Unit:", profit)
    print("Gross Margin:", round(margin, 2), "%")

# ==========================================
# 7. CORRELATION ANALYSIS
# ==========================================

print("\nDemand Correlation:")

print(df[demand_columns].corr().round(3))

print("\n========== EDA COMPLETED ==========")

print("\nGraphs saved in:", output_dir)
