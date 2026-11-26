
"""
Checks for the user-input scheduling tool.

Covers:

- parsing Excel-like tables (aliases, blank rows, validation),
- arrival times respected by every dispatch rule,
- all five rules producing a complete schedule.

Run from the project root:

    $env:PYTHONIOENCODING = "utf-8"
    venv\\Scripts\\python.exe src\\test_simple_tool.py
"""

import sys

import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.job_shop_env import (
    generate_simple_instance, instance_from_table, run_dispatch,
    simple_table_rows
)


def check_parsing():

    frame = pd.DataFrame({
        "Job ID": ["A", "B", "C", "D"],
        "Arrival Time": [0, 0, 2, 4],
        "Burst Time": [3, 5, 2, 4],
        "Priority": [2, 1, 3, 1]
    })

    instance = instance_from_table(frame, n_machines=2, name="Excel test")

    assert instance["n_jobs"] == 4
    assert instance["n_machines"] == 2

    rows = simple_table_rows(instance)
    assert rows[2]["Arrival Time"] == 2.0
    assert rows[1]["Priority"] == 1

    # Column aliases and a missing priority column default to 1.
    aliased = instance_from_table(
        pd.DataFrame({
            "id": [1, 2],
            "release_time": [0, 1],
            "processing time": [4, 2]
        }),
        n_machines=1
    )
    assert [job["priority"] for job in aliased["jobs"]] == [1, 1]

    # A trailing all-blank row is ignored.
    with_blank = pd.concat([
        frame,
        pd.DataFrame([{
            "Job ID": None, "Arrival Time": None,
            "Burst Time": None, "Priority": None
        }])
    ], ignore_index=True)
    assert instance_from_table(with_blank, n_machines=2)["n_jobs"] == 4

    invalid_frames = [
        (pd.DataFrame({"Arrival Time": [-1], "Burst Time": [2]}),
         "arrival"),
        (pd.DataFrame({"Arrival Time": [0], "Burst Time": [0]}),
         "burst"),
        (pd.DataFrame({
            "Arrival Time": [0], "Burst Time": [2], "Priority": [1.5]
        }), "priority"),
        (pd.DataFrame({"Arrival Time": [0]}), "column")
    ]

    for bad_frame, expected in invalid_frames:

        try:
            instance_from_table(bad_frame, n_machines=2)
        except ValueError as error:
            assert expected in str(error).lower(), (expected, error)
        else:
            raise AssertionError(f"Invalid frame accepted: {bad_frame}")

    print("Parsing checks passed.")


def check_rule_schedules():

    instance = generate_simple_instance(
        seed=3,
        n_jobs=12,
        n_machines=3,
        arrivals=[0, 0, 0, 3, 3, 5, 8, 8, 10, 11, 12, 15],
        bursts=[7, 2, 5, 1, 6, 3, 8, 2, 4, 1, 5, 2],
        priorities=[3, 1, 2, 1, 3, 2, 1, 3, 2, 1, 2, 3]
    )

    for rule in ("random", "fcfs", "spt", "sjf", "priority"):

        env = run_dispatch(instance, strategy=rule, seed=1)
        metrics = env.metrics()

        assert metrics["total_operations"] == 12
        assert len(env.schedule) == 12

        for entry in env.schedule:

            job = instance["jobs"][entry["job_id"] - 1]

            assert entry["start"] >= job["arrival"] - 1e-9, (rule, entry)
            assert entry["end"] > entry["start"]

        print(
            f"{rule:9s} makespan={metrics['makespan']:5.1f}"
            f" wait={metrics['mean_waiting_time']:5.2f}"
        )

    print("Rule checks passed.")


if __name__ == "__main__":
    check_parsing()
    check_rule_schedules()
    print("\nAll simple-tool checks passed.")
