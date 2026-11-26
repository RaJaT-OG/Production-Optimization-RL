
"""
Evaluate all scheduling algorithms on unseen job-shop instances.

The same instance is used for every algorithm (paired comparison):

    Random | FCFS | SPT | DQN | PPO

Outputs:

    reports/jobshop_episodes.csv     one row per instance and algorithm
    reports/jobshop_comparison.csv   aggregate table
    reports/figures/jobshop_comparison.png

Run from the project root:

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe evaluation\\evaluate_jobshop.py
"""

import sys

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.job_shop_env import (
    generate_instance, run_agent, run_dispatch
)


N_INSTANCES = 100
SEED_START = 10_000

BASELINES = ("Random", "FCFS", "SPT")
METRICS = (
    "makespan",
    "mean_waiting_time",
    "throughput_per_100",
    "mean_utilization"
)
LOWER_IS_BETTER = {
    "makespan": True,
    "mean_waiting_time": True,
    "throughput_per_100": False,
    "mean_utilization": False
}
SEED = 42


# ==========================================
# MODEL LOADING
# ==========================================

def load_models():

    models = {}

    dqn_path = BASE_DIR / "models" / "jobshop_dqn.zip"

    if dqn_path.exists():

        from agents.masked_dqn import MaskedDQN

        models["DQN"] = MaskedDQN.load(str(dqn_path))
        print("Loaded DQN:", dqn_path)

    else:
        print("DQN model not found, skipping.")

    ppo_path = BASE_DIR / "models" / "jobshop_ppo.zip"

    if ppo_path.exists():

        from sb3_contrib import MaskablePPO

        models["PPO"] = MaskablePPO.load(str(ppo_path))
        print("Loaded PPO:", ppo_path)

    else:
        print("PPO model not found, skipping.")

    return models


# ==========================================
# EVALUATION
# ==========================================

def evaluate():

    models = load_models()

    algorithms = list(BASELINES) + list(models)

    rows = []

    for index in range(N_INSTANCES):

        instance = generate_instance(seed=SEED_START + index)

        for algorithm in algorithms:

            if algorithm in BASELINES:

                env = run_dispatch(
                    instance,
                    strategy=algorithm.lower(),
                    seed=SEED + index
                )

            else:

                env = run_agent(models[algorithm], instance)

            metrics = env.metrics()

            rows.append({
                "instance": index,
                "seed": SEED_START + index,
                "n_jobs": metrics["n_jobs"],
                "n_machines": metrics["n_machines"],
                "algorithm": algorithm,
                "makespan": metrics["makespan"],
                "mean_waiting_time": metrics["mean_waiting_time"],
                "throughput_per_100": metrics["throughput_per_100"],
                "mean_utilization": metrics["mean_utilization"]
            })

        if (index + 1) % 10 == 0:
            print(f"Evaluated {index + 1}/{N_INSTANCES} instances")

    frame = pd.DataFrame(rows)

    reports_dir = BASE_DIR / "reports"
    figures_dir = reports_dir / "figures"

    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = reports_dir / "jobshop_episodes.csv"
    frame.to_csv(episodes_path, index=False)

    comparison = build_comparison(frame, algorithms)

    comparison_path = reports_dir / "jobshop_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    figure_path = figures_dir / "jobshop_comparison.png"
    make_figure(frame, algorithms, figure_path)

    print("\n========== JOB-SHOP COMPARISON (100 unseen instances) ==========")
    print(comparison.to_string(index=False))
    print("================================================================")
    print("\nSaved:", episodes_path)
    print("Saved:", comparison_path)
    print("Saved:", figure_path)

    return frame, comparison


def build_comparison(frame, algorithms, baselines=None):
    """
    Aggregate means, spread and paired improvement against the best
    baseline on each instance.
    """

    baselines = tuple(BASELINES if baselines is None else baselines)

    records = []

    for algorithm in algorithms:

        subset = frame[frame["algorithm"] == algorithm]

        record = {
            "Algorithm": algorithm,
            "Makespan": subset["makespan"].mean(),
            "Makespan std": subset["makespan"].std(),
            "Waiting time": subset["mean_waiting_time"].mean(),
            "Throughput (/100 time)": subset["throughput_per_100"].mean(),
            "Utilization (%)": subset["mean_utilization"].mean()
        }

        records.append(record)

    # PAIRED IMPROVEMENT VS BEST BASELINE PER INSTANCE
    base = frame[frame["algorithm"].isin(baselines)]

    best = {}

    for metric in METRICS:

        grouped = base.groupby("instance")[metric]

        if LOWER_IS_BETTER[metric]:
            best[metric] = grouped.min()
        else:
            best[metric] = grouped.max()

    # MAKESPAN WIN RATE ACROSS ALL ALGORITHMS
    makespan_table = frame.pivot(
        index="instance", columns="algorithm", values="makespan"
    )

    instance_best = makespan_table.min(axis=1)

    win_rates = {
        algorithm: float(
            np.isclose(
                makespan_table[algorithm], instance_best, atol=1e-9
            ).mean()
        ) * 100.0
        for algorithm in algorithms
    }

    improvement_rows = []

    for algorithm in algorithms:

        subset = frame[frame["algorithm"] == algorithm].set_index("instance")

        row = {
            "Algorithm": algorithm,
            "Win rate (%)": win_rates[algorithm]
        }

        for metric in METRICS:

            reference = best[metric].reindex(subset.index)

            if LOWER_IS_BETTER[metric]:
                gain = (reference - subset[metric]) / reference * 100.0
            else:
                gain = (subset[metric] - reference) / reference * 100.0

            row[f"Gain {metric}"] = float(gain.mean())

        improvement_rows.append(row)

    gains = pd.DataFrame(improvement_rows)

    comparison = pd.DataFrame(records).merge(gains, on="Algorithm")

    return comparison


def make_figure(frame, algorithms, path, title=None):

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))

    plot_specs = [
        ("makespan", "Makespan (lower is better)", axes[0, 0]),
        ("mean_waiting_time", "Mean waiting time (lower is better)", axes[0, 1]),
        ("throughput_per_100", "Throughput per 100 time", axes[1, 0]),
        ("mean_utilization", "Machine utilization (%)", axes[1, 1])
    ]

    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

    for metric, title, axis in plot_specs:

        means = [
            frame[frame["algorithm"] == algorithm][metric].mean()
            for algorithm in algorithms
        ]

        errors = [
            frame[frame["algorithm"] == algorithm][metric].std()
            for algorithm in algorithms
        ]

        axis.bar(
            algorithms,
            means,
            yerr=errors,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.6
        )

        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=15)

    figure.suptitle(
        title or "Job-shop scheduling: RL agents vs dispatch-rule baselines"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    evaluate()
