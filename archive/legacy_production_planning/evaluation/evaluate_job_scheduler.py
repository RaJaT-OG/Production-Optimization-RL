
import sys

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.job_scheduling_env import (
    JobSchedulingEnv, run_greedy, run_agent
)

DATA_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"
MODEL_PATH = BASE_DIR / "models" / "job_scheduler_maskableppo.zip"

REPORT_PATH = BASE_DIR / "reports" / "job_scheduler_comparison.csv"
EPISODES_PATH = BASE_DIR / "reports" / "job_scheduler_episodes.csv"
FIGURE_PATH = BASE_DIR / "reports" / "figures" / "job_scheduler_comparison.png"

EPISODES = 100
SEEDS = range(100)

STRATEGIES = [
    "Greedy (earliest due date)",
    "Greedy (profit density)",
    "MaskablePPO agent"
]


# ==========================================
# 1. EPISODE RUNNER
# ==========================================

def run_episode(strategy, jobs, start_day, model):

    if strategy.startswith("MaskablePPO"):
        env = run_agent(
            model, jobs, start_day=start_day,
            data_path=str(DATA_PATH)
        )
    elif strategy.endswith("due date)"):
        env = run_greedy(
            jobs, start_day=start_day, order="edd",
            data_path=str(DATA_PATH)
        )
    else:
        env = run_greedy(
            jobs, start_day=start_day, order="value",
            data_path=str(DATA_PATH)
        )

    return env.summary()


# ==========================================
# 2. EVALUATION PROTOCOL
# ==========================================

def evaluate():

    print()
    print("=" * 60)
    print("JOB SCHEDULER EVALUATION")
    print("=" * 60)
    print(f"Episodes: {EPISODES} (seeds 0..{EPISODES - 1})")
    print("Same job set and factory capacity for every strategy")
    print()

    model = MaskablePPO.load(str(MODEL_PATH))

    rows = []

    template = JobSchedulingEnv(data_path=str(DATA_PATH))

    for seed in SEEDS:

        rng = np.random.default_rng(seed)

        jobs = template.generate_job_set(rng)
        start_day = int(rng.integers(0, len(template.df) - 30))

        for strategy in STRATEGIES:

            result = run_episode(strategy, jobs, start_day, model)

            rows.append({
                "seed": seed,
                "strategy": strategy,
                "n_jobs": result["n_jobs"],
                "profit": result["profit"],
                "on_time_pct": result["on_time_pct"],
                "completed": result["completed"],
                "abandoned": result["abandoned"]
            })

        if (seed + 1) % 20 == 0:
            print(f"  Completed {seed + 1}/{EPISODES} job sets")

    episodes = pd.DataFrame(rows)

    summary = episodes.groupby("strategy").agg(
        mean_profit=("profit", "mean"),
        std_profit=("profit", "std"),
        mean_on_time_pct=("on_time_pct", "mean"),
        mean_completed=("completed", "mean"),
        mean_abandoned=("abandoned", "mean")
    ).reset_index()

    # ==========================================
    # 3. REPORT
    # ==========================================

    reports_dir = BASE_DIR / "reports"
    figures_dir = reports_dir / "figures"

    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    episodes.to_csv(EPISODES_PATH, index=False)
    summary.to_csv(REPORT_PATH, index=False)

    print()
    print("-" * 60)
    print("RESULTS (mean over 100 job sets)")
    print("-" * 60)

    for _, row in summary.iterrows():

        print(
            f"{row['strategy']:<30} "
            f"profit ₹{row['mean_profit']:>10,.0f} "
            f"(+/- {row['std_profit']:,.0f})   "
            f"on-time {row['mean_on_time_pct']:5.1f}%   "
            f"abandoned {row['mean_abandoned']:.1f}"
        )

    # ==========================================
    # 4. FIGURE
    # ==========================================

    plt.figure(figsize=(9, 5))

    positions = np.arange(len(summary))
    colors = ["#888888", "#aaaaaa", "#2e7d32"]

    plt.bar(
        positions,
        summary["mean_profit"],
        yerr=summary["std_profit"],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8
    )

    plt.xticks(positions, [
        "Greedy\n(earliest due date)",
        "Greedy\n(profit density)",
        "MaskablePPO\nagent"
    ])

    plt.ylabel("Total profit per job set (₹)")
    plt.title("Job assignment: RL agent vs greedy baselines (100 job sets)")

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print()
    print("Saved:", REPORT_PATH)
    print("Saved:", EPISODES_PATH)
    print("Saved:", FIGURE_PATH)
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
