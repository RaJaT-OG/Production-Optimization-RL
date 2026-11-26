
"""
Evaluate the user-input scheduling tool on unseen simple instances.

One instance = arrival + burst + priority for every job, with identical
parallel machines. The same instance is used for every algorithm:

    Random | FCFS | SJF | Priority | DQN | PPO

Outputs:

    reports/simple_episodes.csv      one row per instance and algorithm
    reports/simple_comparison.csv    aggregate table
    reports/figures/simple_comparison.png

Run from the project root:

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe evaluation\\evaluate_simple.py
"""

import sys

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "evaluation"))

from environment.job_shop_env import (
    MAX_JOBS, MAX_MACHINES, generate_simple_instance, run_agent,
    run_dispatch
)
from evaluate_jobshop import build_comparison, make_figure


N_INSTANCES = 100
SEED_START = 10_000

BASELINES = ("Random", "FCFS", "SJF", "Priority")
ALGORITHMS = BASELINES + ("DQN", "PPO")
SEED = 42


def load_models():

    models = {}

    dqn_path = BASE_DIR / "models" / "simple_dqn.zip"

    if dqn_path.exists():

        from agents.masked_dqn import MaskedDQN

        models["DQN"] = MaskedDQN.load(str(dqn_path))
        print("Loaded DQN:", dqn_path)

    else:
        print("DQN model not found, skipping.")

    ppo_path = BASE_DIR / "models" / "simple_ppo.zip"

    if ppo_path.exists():

        from sb3_contrib import MaskablePPO

        models["PPO"] = MaskablePPO.load(str(ppo_path))
        print("Loaded PPO:", ppo_path)

    else:
        print("PPO model not found, skipping.")

    return models


def evaluation_instance(index):
    """Deterministic instance from the reserved 10,000+ seed block."""

    seed = SEED_START + index
    rng = np.random.default_rng(seed)

    return generate_simple_instance(
        seed=seed,
        n_jobs=int(rng.integers(5, MAX_JOBS + 1)),
        n_machines=int(rng.integers(1, MAX_MACHINES + 1))
    )


def evaluate():

    models = load_models()

    algorithms = [
        name for name in ALGORITHMS
        if name in BASELINES or name in models
    ]

    rows = []

    for index in range(N_INSTANCES):

        instance = evaluation_instance(index)

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

    episodes_path = reports_dir / "simple_episodes.csv"
    frame.to_csv(episodes_path, index=False)

    comparison = build_comparison(
        frame, algorithms, baselines=BASELINES
    )

    comparison_path = reports_dir / "simple_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    figure_path = figures_dir / "simple_comparison.png"
    make_figure(
        frame,
        algorithms,
        figure_path,
        title=(
            "User-input scheduling tool: RL agents vs "
            "dispatch-rule baselines"
        )
    )

    print(
        "\n========== SIMPLE SCHEDULING COMPARISON "
        "(100 unseen instances) =========="
    )
    print(comparison.to_string(index=False))
    print("=" * 66)
    print("\nSaved:", episodes_path)
    print("Saved:", comparison_path)
    print("Saved:", figure_path)

    return frame, comparison


if __name__ == "__main__":
    evaluate()
