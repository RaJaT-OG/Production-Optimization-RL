# Production Line Optimization using Reinforcement Learning

This project looks at production scheduling from two angles.

The first is an **input-driven scheduling tool**, which is the main thing
you can actually use. You type in jobs (arrival time, burst time and
priority) or upload a spreadsheet, pick how many identical machines you
have, and compare FCFS, SJF, Priority, DQN and PPO on the same jobs. It
runs in a Streamlit dashboard and uses a simple parallel-machine
simulator underneath.

The second is the original **job-shop study**, kept here in full. It uses
3–5 machines and 10–20 jobs, and every job visits several machines in a
fixed order. A Deep Q-Network (DQN) decides which eligible job should run
next. It is compared with Random, FCFS and SPT dispatch rules, plus a PPO
alternative.

**Team:** Shubham (24csu197), Rajat Yadav (24csu170), Dev Kumar (24csu356)

---

## The scheduling tool

### What a job looks like

Each row in the input is one job.

| Column | Required | What it means |
|--------|----------|---------------|
| `Job ID` | No | A label from your sheet. The schedule uses `J1…Jn` in row order. |
| `Arrival Time` | Yes | When the job becomes available (0 or more). |
| `Burst Time` | Yes | How long the job takes on any machine (greater than 0). |
| `Priority` | No | `1` is the highest priority. Use a whole number of 1 or more. If blank, it defaults to 1. |

Column names are not case-sensitive, and common aliases such as
`arrival_time`, `release_time` and `processing time` also work. The
columns can be in any order.

You can enter 1–20 jobs by hand, or upload `.xlsx`, `.xlsm` or `.csv`.
The sidebar also has a downloadable Excel template with an instructions
sheet. The tool checks the input before running anything. Negative
arrival times, zero burst times, non-integer priorities, missing columns
and more than 20 jobs are all rejected, and the error tells you which row
caused the problem.

### Schedulers you can compare

- **FCFS:** whichever job arrived first goes first.
- **SJF:** shortest burst time first. With one operation per job, this is
  the same as SPT.
- **Priority:** lowest priority number first, then earliest arrival.
- **DQN:** a masked Deep Q-Network trained on random instances with 5–20
  jobs, 1–5 machines, arrivals 0–10, bursts 1–9 and priorities 1–3.
- **PPO:** a MaskablePPO alternative trained on the same kind of
  instances.

All five follow the same non-delay rule: a job can start only when a
machine is free. Time jumps to the next arrival or the next machine
completion, whichever comes first. Every scheduler gets exactly the same
instance, so the comparison is fair.

### How the results are measured

- **Makespan:** when the last job finishes. Lower is better.
- **Mean waiting time:** the average of `start − arrival` across jobs.
  Lower is better.
- **Throughput:** jobs completed per 100 time units. Higher is better.
- **Machine utilisation:** busy time divided by makespan, averaged over
  the machines. Higher is better.

### Running it

```bash
venv\Scripts\streamlit.exe run dashboard\app.py
```

Open the **📥 Your data** page from the sidebar. There is a sample file at
`data/example_jobs.csv` if you want to try an upload straight away.

Training the simple-mode agents is optional because the rule-based
schedulers work without them. If you do want to retrain:

```bash
venv\Scripts\python.exe agents\train_dqn_simple.py
venv\Scripts\python.exe agents\train_ppo_simple.py
```

To reproduce the 100-instance comparison:

```bash
venv\Scripts\python.exe evaluation\evaluate_simple.py
```

To check the parser and the dispatch rules:

```bash
venv\Scripts\python.exe src\test_simple_tool.py
```

The comparison uses 100 instances the agents have never seen, with seeds
`10,000–10,099`. Every scheduler gets the same instances. The instances
have 5–20 jobs and 1–5 machines, and arrivals, bursts and priorities come
from the same distribution used in training. The evaluation seeds are
kept separate from the training range (`1,000,000–1,999,999`).

The evaluation writes three files:

- `reports/simple_episodes.csv`: one row per instance and scheduler.
- `reports/simple_comparison.csv`: the aggregate table.
- `reports/figures/simple_comparison.png`: a four-panel comparison.

### Results on 100 unseen instances

These are means over seeds `10,000–10,099`. The win rate is the share of
instances where a scheduler produced the lowest makespan. Ties count as
wins, and with few jobs many makespans are identical.

| Strategy | Makespan | Std | Mean waiting | Throughput (/100 time) | Utilization | Win rate |
|----------|---------:|----:|-------------:|-----------------------:|------------:|---------:|
| Random | 29.36 | 16.68 | 7.10 | 46.35 | 83.12% | 59% |
| FCFS | 29.47 | 16.67 | 7.11 | 45.97 | 82.70% | 50% |
| SJF | 29.81 | 16.53 | **5.12** | 45.18 | 81.49% | 43% |
| Priority | 29.39 | 16.66 | 7.20 | 46.13 | 82.94% | 54% |
| PPO | 29.30 | 16.70 | 7.75 | 46.50 | 83.32% | 61% |
| **DQN** | **28.72** | 16.68 | 9.14 | **47.56** | **85.21%** | **87%** |

The DQN comes out ahead on mean makespan, throughput and utilisation. Its
makespan is 2.0–3.7% shorter than any single rule or PPO. If you compare
it with the best rule chosen separately for each instance, the two are
basically tied: the DQN is 0.003% ahead on the paired comparison. That is
the interesting part. It is not blindly following one rule; its choices
tend to match whichever rule happens to be best for that instance.

The downside is waiting time. SJF is clearly the best there (5.12), while
the DQN is the worst (9.14). The makespan reward encourages starting long
jobs early to keep machines busy, and that leaves short jobs waiting.
Any strategy can win on a single instance, so the aggregate is the real
claim. The dashboard shows the per-instance outcome so you can see when
the DQN loses.

---

## The job-shop study

### What the problem is

| Element | Description |
|---------|-------------|
| Production line | 3–5 machines (`M1 … M5`) |
| Jobs per episode | 10–20 |
| Job structure | 2–4 operations in a fixed order, each on a different machine |
| Processing time | 1–9 time units per operation |
| Decision point | The earliest time a machine and a job are both free |
| Action | Pick an **eligible job**, meaning its next operation can start now |
| Objective | Reduce makespan and waiting time, improve throughput and utilisation |

This is the classic **job-shop scheduling problem**. Jobs do not follow the
same machine order, so the agent has to deal with queues, bottlenecks and
remaining work instead of following one fixed route.

#### Metrics

- **Makespan:** completion time of the last operation. Lower is better.
- **Mean waiting time:** the average of
  `completion − arrival − total processing time`. Lower is better.
- **Throughput:** jobs completed per 100 time units. Higher is better.
- **Machine utilisation:** the share of the makespan each machine spends
  processing, averaged over the active machines. Higher is better.

---

### How it works

#### State

The observation is a fixed-size vector of 309 features, so one policy can
handle any instance size:

- **Machines (5 × 5):** in use, remaining busy time, utilisation so far,
  queued operations, queued workload.
- **Jobs (20 × 13):** active, completed, next machine (one-hot),
  processing time, remaining operations, remaining work, waiting time,
  priority, ready flag.
- **Globals (4):** current time, share of operations completed, number of
  jobs, number of machines.
- **Action mask (20):** which jobs are eligible right now.

#### Action

`Discrete(20)`: pick a job index. The environment masks any job whose next
operation cannot start at the current earliest time, so the agent can
only sample valid jobs.

#### Reward

The reward is paid once, at the end, and scores the whole schedule:

```
reward = −makespan / time_scale
         − 0.005 · Σ waiting / time_scale
         + 0.1
```

`time_scale` is the instance's lower bound. It is the largest of three
things: the workload balance including the latest arrival, the longest job
from its release to its finish, and 1. That keeps rewards comparable
across instance sizes. A positive `REWARD_SCALE = 10` multiplies the
reward so Q-value differences are large enough for stable DQN updates. It
does not change the optimal policy.

The environment also has a dense reward option
(`reward_mode="increment"`) that rewards the negative change in the
partial makespan at every step. In practice that myopic signal pushed the
agent toward short jobs, so the reported DQN uses the terminal reward.

#### Algorithms

- **DQN (primary):** `agents/masked_dqn.py`. `sb3-contrib` has
  `MaskablePPO` but no `MaskableDQN`, so masking was added to the standard
  SB3 DQN. The Q-network sets invalid actions to `−1e9` before action
  selection and before the Bellman target, and epsilon-greedy exploration
  samples only eligible jobs.
- **Q-network:** a shared per-job encoder. Each job is embedded together
  with the state of the machine its next operation needs, so the network
  compares jobs directly instead of learning slot-specific weights. That
  makes it permutation-equivariant and lets it handle any instance size.
- **Warm start:** the Q-network is first trained by supervised imitation
  of the **most-work-remaining (MWKR)** rule on 400 demonstration
  instances, then fine-tuned with DQN. A behaviour-cloning side loss
  (`DemonstrationMaskedDQN`) stops the fine-tuned policy from collapsing
  back to short-sighted dispatching.
- **PPO (alternative):** `sb3-contrib` `MaskablePPO`, using the
  environment's `action_masks()`.
- **Baselines:** Random, FCFS (first-come first-served) and SPT (shortest
  processing time). All are restricted to eligible jobs.

MWKR also appears in the analysis as a strong makespan reference. It is a
dispatch rule, not a learned agent.

---

## Project structure

```
Production-Optimization-RL/
│
├── environment/
│   └── job_shop_env.py           # Simple + job-shop simulator, parser, rules
│
├── agents/
│   ├── masked_dqn.py             # Masked DQN (primary algorithm)
│   ├── train_dqn_simple.py       # DQN for the input-driven tool
│   ├── train_ppo_simple.py       # PPO for the input-driven tool
│   ├── train_dqn_jobshop.py      # DQN training + validation
│   └── train_ppo_jobshop.py      # PPO alternative training
│
├── evaluation/
│   ├── evaluate_simple.py        # 100-instance tool evaluation
│   └── evaluate_jobshop.py       # 100-instance job-shop evaluation
│
├── src/
│   ├── build_processed_data.py   # Cleans raw data + builds features
│   ├── test_simple_tool.py
│   └── test_jobshop_environment.py
│
├── dashboard/app.py              # Streamlit: user data + job-shop demo
├── data/
│   ├── example_jobs.csv          # Sample input for the user-data page
│   ├── raw/
│   │   └── production_data.csv   # Original daily production dataset
│   └── processed/
│       ├── production_data_clean.csv
│       ├── production_features.csv
│       └── summary.json
├── models/                       # simple_dqn/ppo + jobshop_dqn/ppo
├── reports/                      # CSV results, logs and figures
├── logs/                         # TensorBoard training curves
│
├── archive/
│   └── legacy_production_planning/  # Archived V1–V6 work (reference only)
│
├── requirements.txt
├── HOW_TO_RUN.md                 # Friend-facing setup and run guide
└── README.md
```

### The processed datasets

The old production data now has two derived versions in `data/processed/`:

| File | Rows × columns | What is in it |
| --- | --- | --- |
| `production_data_clean.csv` | 3,000 × 20 | Validated, typed, deduplicated and day-sorted raw data |
| `production_features.csv` | 3,000 × 39 | The clean data plus demand, economics, resource and utilisation features |
| `summary.json` | — | Row counts, day range, quality checks and output paths |

The feature file adds these groups:

- **Demand:** `total_demand`, `demand_share_A/B/C`, `demand_7day_avg`
- **Economics:** `total_revenue`, `total_production_cost`,
  `gross_margin`, `gross_margin_pct`
- **Resources:** `total_material_required`,
  `total_machine_hours_required`
- **Utilisation:** `raw_material_utilization`,
  `machine_hours_utilization`, `budget_utilization`,
  `storage_utilization`
- **Headroom:** `material_headroom`, `machine_hours_headroom`,
  `budget_headroom`
- **Flag:** `capacity_feasible`, which is true when all three resource
  utilisation ratios are at or below 1.0

You can rebuild both files from the raw data at any time:

```bash
venv\Scripts\python.exe src\build_processed_data.py
```

On the current data, 621 of the 3,000 days can produce the full recorded
demand within the available raw material, machine hours and budget. The
rest hit at least one limit, usually machine hours.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

---

## Training and evaluation (job-shop)

Train the primary DQN agent. It uses the MWKR warm start and then 400,000
fine-tuning steps, which takes about 15 minutes on a CPU:

```bash
venv\Scripts\python.exe agents\train_dqn_jobshop.py
```

Train the PPO alternative (200,000 steps):

```bash
venv\Scripts\python.exe agents\train_ppo_jobshop.py
```

Run the paired evaluation on 100 unseen instances:

```bash
venv\Scripts\python.exe evaluation\evaluate_jobshop.py
```

Check the environment:

```bash
venv\Scripts\python.exe src\test_jobshop_environment.py
```

Launch the dashboard:

```bash
venv\Scripts\streamlit.exe run dashboard\app.py
```

Watch training in TensorBoard:

```bash
tensorboard --logdir logs/
```

On Windows, prefix long runs with `$env:PYTHONIOENCODING = "utf-8"` so the
console can render the ₹ symbol and progress bars properly.

---

## Evaluation protocol (job-shop)

- **Evaluation instances:** 100 unseen instances with seeds `10,000–10,099`,
  shared by every algorithm. Every comparison is paired because all
  algorithms solve exactly the same jobs.
- **Training instances:** random instances from a separate reserved seed
  range (`1,000,000–1,999,999`), so the evaluation instances are never
  seen during training.
- **Warm-start demonstrations:** seeds `2,000,000` and above.
- **Model selection:** 10 validation instances (seeds `9,000–9,009`) during
  training, plus 30 more (seeds `20,000–20,029`) for tuning. Neither set is
  part of the final 100-instance report.
- **Reporting:** means over the 100 instances, paired improvement against
  the best baseline on each instance, and makespan win rate.

The evaluation writes:

- `reports/jobshop_episodes.csv`: one row per instance and algorithm.
- `reports/jobshop_comparison.csv`: the aggregate table.
- `reports/figures/jobshop_comparison.png`: a four-panel comparison.

---

## Job-shop results

These are means over **100 unseen instances** (seeds `10,000–10,099`),
shared by every algorithm. Lower is better for makespan and waiting time;
higher is better for throughput, utilisation and win rate. Win rate is the
share of instances where a strategy had the lowest makespan.

| Strategy | Makespan | Std | Mean waiting | Throughput (/100 time) | Utilization | Win rate |
|----------|---------:|----:|-------------:|-----------------------:|------------:|---------:|
| Random | 70.44 | 15.78 | 29.24 | 20.98 | 75.49% | 37% |
| FCFS | 72.39 | 14.50 | 27.22 | 20.28 | 73.09% | 20% |
| SPT | 70.76 | 15.45 | 22.78 | 20.83 | 74.98% | 30% |
| PPO (masked) | 70.55 | 15.26 | 28.21 | 20.89 | 75.15% | 28% |
| **DQN (primary)** | **66.71** | **14.59** | 35.79 | **22.13** | **79.47%** | **72%** |

The DQN has the lowest makespan, the highest throughput and the highest
machine utilisation, and it wins 72 of the 100 instances. On the paired
comparison it improves makespan by 1.2% over the best baseline on each
instance. That number is small because a simple rule is already close to
optimal on many instances. The win rate and the 5.3% gap to Random's mean
tell the clearer story.

For context, here are two classical makespan-oriented dispatch rules on
the same 100 instances. These are not learned agents.

| Reference rule | Makespan | Mean waiting | Throughput (/100 time) | Utilization |
|----------------|---------:|-------------:|-----------------------:|------------:|
| MWKR (most work remaining) | 66.49 | 35.86 | 22.22 | 79.77% |
| MOPNR (most operations remaining) | 66.04 | 33.48 | 22.38 | 80.33% |

The learned DQN lands within 0.3 makespan units of MWKR and 0.7 of MOPNR.
In other words, it learns to behave like the strong makespan-oriented
rules while still beating Random, FCFS, SPT and PPO.

Full per-instance data is in `reports/jobshop_episodes.csv`. The aggregate
table is in `reports/jobshop_comparison.csv`, and the figure is at
`reports/figures/jobshop_comparison.png`.

## What we learned

1. **The learned agent beats every standard rule on makespan.** DQN reaches
   66.71 against 70.44 (Random), 70.76 (SPT), 72.39 (FCFS) and 70.55
   (PPO). It also has a 72% makespan win rate, the best throughput and the
   best utilisation.
2. **There is a real makespan / waiting-time trade-off.** The DQN's mean
   waiting time (35.79) is worse than SPT (22.78). Makespan and waiting
   time reward opposite decisions. Starting long jobs early keeps the
   bottleneck busy, but it makes short jobs queue. If the objective were
   average waiting time, SPT would be the right choice.
3. **A from-scratch DQN learns the wrong rule.** With no warm start, the
   Q-values correlated *negatively* with remaining work (`corr = −0.42`),
   so the agent behaved like SPT. The dense `"increment"` reward is myopic
   because it rewards short operations locally. The terminal reward alone
   was also unstable, with a 76.90 makespan on validation.
4. **The MWKR warm start was the decisive change.** Supervised imitation
   of MWKR for 60 epochs reached 91.7% decision accuracy and matched MWKR
   exactly before fine-tuning (69.13 vs 69.13 on 30 validation instances).
   Fine-tuning without regularisation destroyed that policy, so we added a
   behaviour-cloning side loss (`DemonstrationMaskedDQN`). With it,
   fine-tuning stayed stable and finished at 68.10 on the 10-instance
   validation set.
5. **The per-job encoder helps waiting time, not makespan.** Replacing the
   flat MLP with a shared per-job encoder gave the best waiting time among
   the non-warm-started variants (22.71), but it did not fix the makespan
   objective on its own.
6. **PPO does not beat the baselines.** MaskablePPO lands at 70.55, close
   to Random. Action masking removes invalid actions, but it does not solve
   the long-horizon credit assignment problem by itself.

### Ablation

| Q-network | Reward | Warm start | Evaluation set | Makespan |
|-----------|--------|------------|----------------|---------:|
| Flat MLP [256, 256] | increment, γ = 0.99 | none | 100 instances | 73.25 |
| Flat MLP [256, 256] | increment, γ = 1.0 | none | 100 instances | 70.57 |
| Per-job encoder | increment, γ = 1.0 | none | 100 instances | 71.82 |
| Per-job encoder | terminal, γ = 1.0 | none | 10 validation | 76.90 |
| Per-job encoder | terminal, γ = 1.0 | MWKR + BC | 100 instances | **66.71** |

## Implementation notes

- Both simulators, the dispatch rules, the metrics and the input-table
  parser live in `environment/job_shop_env.py`. The project does not depend
  on `jobshoplib`.
- The user-input tool reuses the same masked DQN and observation encoder in
  `simple=True` mode. Every job has one flexible operation, arrival times
  are respected, and the instance size is still capped at 20 jobs × 5
  machines. The simple DQN was fine-tuned for 300,000 steps after the same
  MWKR warm start. The simple PPO was trained for 200,000 steps.
- Excel and CSV input goes through `instance_from_table`. It matches column
  aliases, ignores blank rows, and reports validation errors with the
  spreadsheet row number. `openpyxl` is needed for `.xlsx` files.
- `MaskedDQN` is a small extension of Stable-Baselines3's DQN. The action
  mask is appended to the observation, so it travels with every transition
  in the replay buffer. That keeps the Bellman target consistent with the
  behaviour policy.
- `DemonstrationMaskedDQN` adds one behaviour-cloning gradient step per DQN
  update on the stored MWKR demonstrations. This is the
  demonstration-regularised fine-tuning that keeps the warm-started policy
  stable.
- The Q-network uses a shared per-job encoder: job features, the features
  of the machine needed by the next operation, globals and eligibility. It
  is permutation-equivariant and supports any number of jobs up to 20.
- The FastAPI service and Docker deployment mentioned in the original
  project plan are **not implemented** here. The demo is a Streamlit
  dashboard.
- The earlier experiments from this project, the daily production planning
  and the job-to-day assignment module (V1–V6), are **archived** under
  `archive/legacy_production_planning/`. They are reference-only and are
  not part of the main pipeline. Their absolute imports assume the old
  folder layout.
