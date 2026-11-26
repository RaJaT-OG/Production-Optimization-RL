
"""
Sanity checks for the job-shop environment.

Run from the project root:

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe src\\test_jobshop_environment.py
"""

import sys

from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.job_shop_env import (
    DISPATCH_RULES, OBS_SIZE, JobShopEnv, generate_instance, preset_instance,
    run_dispatch
)


def check_observation(env, observation):

    assert observation.shape == (OBS_SIZE,), observation.shape
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert observation.min() >= 0.0
    assert observation.max() <= 1.0

    mask = env.action_masks()
    stored_mask = observation[-env.n_actions:] >= 0.5

    assert np.array_equal(mask, stored_mask), "mask drift in observation"
    assert mask.any(), "no eligible jobs while pending operations remain"


def check_schedule(env, instance):

    metrics = env.metrics()
    schedule = env.schedule

    assert metrics["makespan"] > 0
    assert len(schedule) == metrics["total_operations"]
    assert 0.0 <= metrics["mean_utilization"] <= 100.0
    assert metrics["mean_waiting_time"] >= -1e-9
    assert metrics["throughput_per_100"] > 0

    # Every operation appears exactly once.
    seen = set()

    for entry in schedule:
        key = (entry["job_id"], entry["operation"])
        assert key not in seen, f"duplicate operation {key}"
        seen.add(key)

    expected = sum(len(job["operations"]) for job in instance["jobs"])
    assert len(seen) == expected

    # Makespan is the last completion time.
    last_end = max(entry["end"] for entry in schedule)
    assert abs(last_end - metrics["makespan"]) < 1e-9


def main():

    instance = generate_instance(seed=1234)

    assert instance["n_machines"] in (3, 4, 5)
    assert 10 <= instance["n_jobs"] <= 20

    env = JobShopEnv()
    observation, _ = env.reset(seed=0, options={"instance": instance})

    check_observation(env, observation)

    for strategy in DISPATCH_RULES:

        env = run_dispatch(instance, strategy=strategy, seed=0)
        check_schedule(env, instance)

        print(
            f"{strategy:6s} makespan={env.metrics()['makespan']:6.1f}"
            f" wait={env.metrics()['mean_waiting_time']:6.1f}"
            f" util={env.metrics()['mean_utilization']:5.1f}%"
        )

    for name in ("Small — 10 jobs / 3 machines",):

        preset = preset_instance(name)
        env = run_dispatch(preset, strategy="spt", seed=0)
        check_schedule(env, preset)

    print("\nAll job-shop environment checks passed.")


if __name__ == "__main__":
    main()
