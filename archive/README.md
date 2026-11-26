# Archive: the old V1–V6 work

This folder is a record of the earlier experiments. Nothing here is part
of the current tool, and nothing here is needed to run the dashboard.

The active code lives in the project root and uses only:

- `environment/job_shop_env.py`
- `agents/masked_dqn.py` and the four `train_*` scripts
- `evaluation/evaluate_simple.py` and `evaluation/evaluate_jobshop.py`
- `dashboard/app.py`

## What's inside

`archive/legacy_production_planning/` keeps the old layout:

| Folder | What it has |
| --- | --- |
| `agents/` | V1–V6 trainers, including the daily production and job-to-day assignment agents |
| `environment/` | The `factory_env*.py` variants and `job_scheduling_env.py` |
| `evaluation/` | The V1–V6 evaluation, comparison and diagnostic scripts |
| `baseline/` | The rule-based baseline used by the old evaluations |
| `src/` | Data generation, analysis and environment test scripts |
| `models/` | Old PPO/DQN checkpoints for V1–V6. The dashboard does not use them |
| `reports/` | Old CSV results and figures |
| `logs/` | Old TensorBoard runs |
| `notebooks/` | The early exploration notebook |

The original raw dataset is back in the active project at
`data/raw/production_data.csv`. It is not duplicated here.

## One thing to know before reusing this code

The old scripts use absolute imports like these:

```python
from environment.factory_env import FactoryEnv
from baseline.baseline_strategy import choose_action
```

Those imports assumed the legacy files were still in the live project
folders. If you want to run one of these scripts again, either copy the
files it needs back beside the current code, or add the archive folder to
`sys.path` and adjust the imports. The current dashboard and evaluations
don't depend on anything in this archive.
