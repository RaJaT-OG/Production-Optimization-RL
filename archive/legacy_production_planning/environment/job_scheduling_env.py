
import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
from pathlib import Path


class JobSchedulingEnv(gym.Env):
    """
    Job assignment environment.

    A set of production jobs (customer orders for products A, B, C)
    is given upfront and must be assigned to production days inside
    a 30-day horizon. Each job needs machine hours, raw material and
    production budget on the day it runs.

    Action space: assign the current job to production day 0..29,
    or abandon it (action 30). Jobs finished by their due day earn
    full revenue; late jobs lose 5% of revenue per day late; jobs
    that do not fit anywhere are abandoned and earn nothing.

    Product economics are identical to the rest of the project.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path=None, horizon=30,
                 min_jobs=20, max_jobs=65,
                 min_qty=10, max_qty=80):

        super().__init__()

        # LOAD DATASET
        if data_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            data_path = base_dir / "data" / "raw" / "production_data.csv"

        self.df = pd.read_csv(data_path)

        self.horizon = horizon
        self.min_jobs = min_jobs
        self.max_jobs = max_jobs
        self.min_qty = min_qty
        self.max_qty = max_qty

        # PRODUCT-SPECIFIC JOB SIZE CAPS
        # Keeps every generated job small enough to fit on a
        # typical day: A needs 2h/unit, B 1.5h/unit, C 1h/unit
        # with about 100 machine hours per day.
        self.max_qty_by_product = np.array([
            min(max_qty, 50),
            min(max_qty, 66),
            max_qty
        ])

        # PRODUCT CONFIGURATION (same as the base factory env)
        self.products = ["A", "B", "C"]

        self.prices = np.array([500, 400, 300])
        self.costs = np.array([250, 180, 120])

        self.material_required = np.array([3, 2, 1])
        self.hours_required = np.array([2, 1.5, 1])

        self.late_penalty_rate = 0.05
        self.invalid_action_penalty = 2000

        # DAILY CAPACITY COLUMNS (hours, material, budget)
        self.capacity_columns = [
            "machine_hours_capacity",
            "raw_material_capacity",
            "production_budget"
        ]

        # ACTION SPACE: day 0..horizon-1, or abandon
        self.n_days = horizon
        self.n_actions = horizon + 1

        self.action_space = spaces.Discrete(self.n_actions)

        # OBSERVATION SPACE
        # [current job: product one-hot(3), quantity, machine hours,
        #  raw material, production cost, due day, revenue, margin,
        #  remaining capacity ratios for every day (3 per day),
        #  assignment progress, number of jobs]

        obs_size = 10 + 3 * self.n_days + 2

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # INITIAL VARIABLES
        self.jobs = []
        self.capacity = np.zeros((self.n_days, 3))
        self.remaining = np.zeros((self.n_days, 3))
        self.usage = np.zeros((self.n_days, 3))

        self.job_idx = 0
        self.start_day = 0

        self.total_profit = 0.0
        self.total_revenue = 0.0
        self.on_time_count = 0
        self.completed_count = 0
        self.abandoned_count = 0
        self.schedule = []

    # ==========================================
    # JOB SET GENERATION
    # ==========================================

    def generate_job_set(self, rng, rush_fraction=0.0):
        """
        Random job set used for training and evaluation.

        rush_fraction: share of jobs with an early deadline
        (due within the first third of the horizon).
        """

        n_jobs = int(rng.integers(self.min_jobs, self.max_jobs + 1))

        jobs = []

        for _ in range(n_jobs):

            product = int(rng.integers(0, 3))

            product_max = int(self.max_qty_by_product[product])

            quantity = int(rng.integers(self.min_qty, product_max + 1))

            if rng.random() < rush_fraction:
                due_day = int(rng.integers(2, self.horizon // 3 + 1))
            else:
                due_day = int(rng.integers(2, self.horizon + 1))

            jobs.append({
                "product": product,
                "quantity": quantity,
                "due_day": due_day
            })

        return jobs

    # ==========================================
    # JOB ECONOMICS
    # ==========================================

    def prepare_jobs(self, jobs):
        """
        Validate job entries and derive machine hours, raw
        material, cost, revenue and margin for every job.
        """

        cleaned = []

        for job in jobs:

            product = int(job["product"])
            quantity = int(max(1, job["quantity"]))
            due_day = int(min(max(1, job["due_day"]), self.horizon))

            cleaned.append({
                "product": min(max(product, 0), 2),
                "quantity": quantity,
                "due_day": due_day
            })

        for job in cleaned:

            p = job["product"]
            q = job["quantity"]

            job["hours"] = q * self.hours_required[p]
            job["material"] = q * self.material_required[p]
            job["cost"] = q * self.costs[p]
            job["revenue"] = q * self.prices[p]
            job["margin"] = job["revenue"] - job["cost"]

        return cleaned

    # ==========================================
    # DAILY CAPACITIES
    # ==========================================

    def load_capacity(self, start_day):

        max_start = len(self.df) - self.horizon
        self.start_day = int(min(max(0, start_day), max_start))

        window = self.df.iloc[
            self.start_day:self.start_day + self.horizon
        ]

        self.capacity = np.column_stack([
            window[column].to_numpy(dtype=float)
            for column in self.capacity_columns
        ])

        self.remaining = self.capacity.copy()
        self.usage = np.zeros((self.n_days, 3))

    # ==========================================
    # CREATE STATE
    # ==========================================

    def get_state(self):

        if self.job_idx >= len(self.jobs):
            job = {
                "product": 0, "quantity": 0, "hours": 0.0,
                "material": 0.0, "cost": 0.0, "due_day": self.horizon,
                "revenue": 0.0, "margin": 0.0
            }
        else:
            job = self.jobs[self.job_idx]

        onehot = np.zeros(3)
        onehot[job["product"]] = 1.0

        capacity_ratios = self.remaining / np.maximum(
            self.capacity, 1e-9
        )

        state = np.concatenate([

            onehot,

            np.array([
                job["quantity"] / 100.0,
                job["hours"] / 200.0,
                job["material"] / 250.0,
                job["cost"] / 25000.0,
                job["due_day"] / float(self.horizon),
                job["revenue"] / 45000.0,
                job["margin"] / 25000.0
            ]),

            capacity_ratios.flatten(),

            np.array([
                self.job_idx / float(max(len(self.jobs), 1)),
                len(self.jobs) / 70.0
            ])

        ])

        return np.clip(state, 0.0, 1.0).astype(np.float32)

    # ==========================================
    # ACTION MASK (for MaskablePPO)
    # ==========================================

    def action_masks(self):

        job = self.jobs[self.job_idx]

        fits = (
            (job["hours"] <= self.remaining[:, 0])
            & (job["material"] <= self.remaining[:, 1])
            & (job["cost"] <= self.remaining[:, 2])
        )

        # Abandoning a job is always allowed
        masks = np.append(fits, True)

        return masks.astype(bool)

    # ==========================================
    # RESET ENVIRONMENT
    # ==========================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        options = options or {}

        if "jobs" in options:

            # Custom job set (dashboard / evaluation)
            jobs = self.prepare_jobs(options["jobs"])

            sort_mode = options.get("sort", "none")
            start_day = options.get("start_day", 1500)

        else:

            # Random job set (training)
            jobs = self.prepare_jobs(
                self.generate_job_set(self.np_random)
            )

            sort_mode = options.get("sort", "due")
            start_day = int(
                self.np_random.integers(0, len(self.df) - self.horizon)
            )

        if sort_mode == "due":
            jobs.sort(key=lambda j: (
                j["due_day"], j["product"], j["quantity"]
            ))

        for i, job in enumerate(jobs):
            job["job_id"] = i + 1

        self.jobs = jobs
        self.load_capacity(start_day)

        self.job_idx = 0
        self.total_profit = 0.0
        self.total_revenue = 0.0
        self.on_time_count = 0
        self.completed_count = 0
        self.abandoned_count = 0
        self.schedule = []

        return self.get_state(), {}

    # ==========================================
    # STEP FUNCTION
    # ==========================================

    def step(self, action):

        job = self.jobs[self.job_idx]
        action = int(action)

        scheduled_day = None
        on_time = None
        realized = 0.0
        contribution = 0.0

        if action < self.n_days:

            feasible = (
                job["hours"] <= self.remaining[action, 0]
                and job["material"] <= self.remaining[action, 1]
                and job["cost"] <= self.remaining[action, 2]
            )

            if feasible:

                # CONSUME DAILY CAPACITY
                self.remaining[action, 0] -= job["hours"]
                self.remaining[action, 1] -= job["material"]
                self.remaining[action, 2] -= job["cost"]

                self.usage[action] += [
                    job["hours"], job["material"], job["cost"]
                ]

                scheduled_day = action + 1

                if scheduled_day <= job["due_day"]:
                    realized = job["revenue"]
                    on_time = True
                else:
                    days_late = scheduled_day - job["due_day"]
                    factor = max(
                        1.0 - self.late_penalty_rate * days_late, 0.0
                    )
                    realized = job["revenue"] * factor
                    on_time = False

                contribution = realized - job["cost"]
                self.completed_count += 1

                if on_time:
                    self.on_time_count += 1

            else:

                # UNMASKED INFEASIBLE ACTION (safety net)
                contribution = -self.invalid_action_penalty

        else:

            # ABANDON THE JOB: no revenue, no cost
            self.abandoned_count += 1

        reward = contribution / 1000.0

        self.total_profit += contribution
        self.total_revenue += realized

        self.schedule.append({
            "job_id": int(job["job_id"]),
            "product": self.products[job["product"]],
            "quantity": int(job["quantity"]),
            "hours": float(job["hours"]),
            "due_day": int(job["due_day"]),
            "scheduled_day": (
                int(scheduled_day) if scheduled_day is not None else None
            ),
            "on_time": on_time,
            "revenue": float(realized),
            "profit": float(contribution)
        })

        self.job_idx += 1

        terminated = self.job_idx >= len(self.jobs)
        truncated = False

        # INFORMATION
        info = {
            "job_id": job["job_id"],
            "scheduled_day": scheduled_day,
            "on_time": on_time,
            "profit": float(contribution),
            "total_profit": float(self.total_profit),
            "total_revenue": float(self.total_revenue),
            "on_time_count": self.on_time_count,
            "completed_count": self.completed_count,
            "abandoned_count": self.abandoned_count
        }

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
            info
        )

    # ==========================================
    # EPISODE SUMMARY
    # ==========================================

    def summary(self):

        n_jobs = len(self.jobs)

        return {
            "profit": float(self.total_profit),
            "revenue": float(self.total_revenue),
            "on_time_pct": 100.0 * self.on_time_count / max(n_jobs, 1),
            "completed": self.completed_count,
            "abandoned": self.abandoned_count,
            "n_jobs": n_jobs
        }

    # ==========================================
    # RENDER ENVIRONMENT
    # ==========================================

    def render(self):

        print("\n========== JOB SCHEDULE ==========")
        print("Jobs:", len(self.jobs))
        print("Total profit:", round(self.total_profit, 2))
        print("On-time jobs:", self.on_time_count)
        print("==================================")


# ==========================================
# GREEDY BASELINE (earliest feasible day)
# ==========================================

def run_greedy(jobs, start_day=1500, order="edd", data_path=None):
    """
    Schedule jobs with a greedy heuristic.

    order="edd":   jobs in earliest-due-date order (classic rule).
    order="value": jobs in margin-per-machine-hour order
                   (most profitable work first).

    Every job is placed on the earliest day with enough remaining
    capacity; jobs that fit nowhere are abandoned.
    """

    env = JobSchedulingEnv(data_path=data_path)

    if order == "value":

        def sort_key(j):
            p, q = j["product"], j["quantity"]
            margin = q * (env.prices[p] - env.costs[p])
            hours = q * env.hours_required[p]
            return -(margin / max(hours, 1e-9))

    else:

        def sort_key(j):
            return (j["due_day"], j["product"], j["quantity"])

    ordered = sorted(jobs, key=sort_key)

    env.reset(seed=0, options={
        "jobs": ordered,
        "start_day": start_day,
        "sort": "none"
    })

    done = False

    while not done:

        masks = env.action_masks()
        feasible_days = np.where(masks[:env.n_days])[0]

        action = int(feasible_days[0]) if len(feasible_days) else env.n_days

        _, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    return env


# ==========================================
# TRAINED AGENT
# ==========================================

def run_agent(model, jobs, start_day=1500, data_path=None):
    """
    Schedule jobs with a trained MaskablePPO policy.
    Jobs are presented in due-date order, as in training.
    """

    env = JobSchedulingEnv(data_path=data_path)

    observation, _ = env.reset(seed=0, options={
        "jobs": jobs,
        "start_day": start_day,
        "sort": "due"
    })

    done = False

    while not done:

        mask = env.action_masks()

        action, _ = model.predict(
            observation,
            action_masks=mask,
            deterministic=True
        )

        observation, _, terminated, truncated, _ = env.step(int(action))

        done = terminated or truncated

    return env
