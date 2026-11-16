
"""
Job-shop production line environment.

The production line contains 3-5 machines. A set of 10-20 jobs must
be processed, and every job is a sequence of operations on specific
machines. At each decision point the agent chooses one eligible job:
a job whose next operation can start now on a free machine.

This follows the project specification:

    State    machine availability, utilization, queue length,
             processing time, waiting time, priority, remaining ops
    Action   select the next eligible job
    Reward   penalise waiting/idle time, then reward a short makespan
    Metrics  makespan, waiting time, throughput, utilization

The observation is a fixed-size vector (max 5 machines, max 20 jobs)
so one policy can be trained on instances of different sizes. The
action mask is appended to the observation, which lets the custom
masked DQN read it directly from each transition.

The same simulator supports a simple input-driven mode
(``JobShopEnv(simple=True)``): every job has one flexible operation
that can run on any machine, with a user-supplied arrival time, burst
time and priority. The dashboard's "Your data" page uses this mode.

Run a baseline:

    env = run_dispatch(instance, strategy="spt")

Run a trained agent:

    env = run_agent(model, instance)
"""

import gymnasium as gym
import numpy as np

from gymnasium import spaces


# ==========================================
# FIXED ENVIRONMENT SIZE
# ==========================================

MAX_MACHINES = 5
MAX_JOBS = 20
MAX_OPERATIONS = 4
MAX_PROCESSING_TIME = 9

MACHINE_FEATURES = 5
JOB_FEATURES = 13
GLOBAL_FEATURES = 4

OBS_SIZE = (
    MAX_MACHINES * MACHINE_FEATURES
    + MAX_JOBS * JOB_FEATURES
    + GLOBAL_FEATURES
    + MAX_JOBS
)

N_ACTIONS = MAX_JOBS

DISPATCH_RULES = ("random", "fcfs", "spt", "sjf", "priority")

# Reward multiplier. Scaling by a positive constant leaves the optimal
# policy unchanged but makes Q-value differences easier to learn.
REWARD_SCALE = 10.0


# ==========================================
# INSTANCE GENERATION
# ==========================================

def generate_instance(seed=None, n_machines=None, n_jobs=None, name=None):
    """
    Build one random job-shop instance.

    Every job visits 2-4 distinct machines in a random order. Processing
    times are integers from 1 to 9, matching the scale of the example in
    the project document (M1:3, M2:5, M3:2).
    """

    rng = np.random.default_rng(seed)

    if n_machines is None:
        n_machines = int(rng.integers(3, MAX_MACHINES + 1))

    if n_jobs is None:
        n_jobs = int(rng.integers(10, MAX_JOBS + 1))

    n_machines = int(np.clip(n_machines, 3, MAX_MACHINES))
    n_jobs = int(np.clip(n_jobs, 10, MAX_JOBS))

    max_ops = min(MAX_OPERATIONS, n_machines)

    jobs = []

    for job_index in range(n_jobs):

        n_ops = int(rng.integers(2, max_ops + 1))

        machines = rng.choice(
            n_machines, size=n_ops, replace=False
        )

        operations = [
            {
                "machine": int(machine),
                "time": int(rng.integers(1, MAX_PROCESSING_TIME + 1))
            }
            for machine in machines
        ]

        jobs.append({
            "job_id": job_index + 1,
            "priority": int(rng.integers(1, 4)),
            "arrival": 0,
            "operations": operations
        })

    return {
        "name": name or "Random instance",
        "seed": seed,
        "n_machines": n_machines,
        "n_jobs": n_jobs,
        "jobs": jobs
    }


def generate_simple_instance(
    seed=None,
    n_jobs=10,
    n_machines=3,
    arrivals=None,
    bursts=None,
    priorities=None,
    name=None
):
    """
    Build a simple parallel-machine instance.

    Every job has exactly one operation, and that operation can run on
    any machine. This is the arrival + burst + priority model used by
    the input-driven dashboard. Priority 1 is the highest priority.

    Supply ``arrivals``, ``bursts`` and ``priorities`` to use the
    user's own data; leave them as ``None`` to sample a random
    instance.
    """

    rng = np.random.default_rng(seed)

    n_jobs = int(np.clip(n_jobs, 1, MAX_JOBS))
    n_machines = int(np.clip(n_machines, 1, MAX_MACHINES))

    if arrivals is None:
        arrivals = rng.integers(0, 11, size=n_jobs)
    if bursts is None:
        bursts = rng.integers(1, MAX_PROCESSING_TIME + 1, size=n_jobs)
    if priorities is None:
        priorities = rng.integers(1, 4, size=n_jobs)

    arrivals = [float(value) for value in arrivals]
    bursts = [float(value) for value in bursts]
    priorities = [int(value) for value in priorities]

    if not (
        len(arrivals) == len(bursts) == len(priorities) == n_jobs
    ):
        raise ValueError(
            "arrivals, bursts and priorities must have n_jobs entries"
        )

    jobs = []

    for index in range(n_jobs):

        jobs.append({
            "job_id": index + 1,
            "priority": priorities[index],
            "arrival": arrivals[index],
            "operations": [
                {"machine": None, "time": bursts[index]}
            ]
        })

    return {
        "name": name or "Simple instance",
        "seed": seed,
        "n_machines": n_machines,
        "n_jobs": n_jobs,
        "jobs": jobs
    }


# ==========================================
# USER INPUT TABLE (EXCEL / MANUAL ENTRY)
# ==========================================

SIMPLE_COLUMN_ALIASES = {
    "job": ("job", "job_id", "job_no", "job_number", "id"),
    "arrival": (
        "arrival", "arrival_time", "release", "release_time", "ready_time"
    ),
    "burst": (
        "burst", "burst_time", "processing_time", "process_time",
        "duration", "processing"
    ),
    "priority": ("priority", "prio", "priority_level")
}


def _normalise_column(name):
    """Make column matching forgiving: case, spaces and underscores."""

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _pick_column(record, field):
    """Return the first alias present in one row, or None."""

    for alias in SIMPLE_COLUMN_ALIASES[field]:
        if alias in record:
            return record[alias]

    return None


def _is_blank(value):
    """True for empty Excel cells and NaN values."""

    if value is None:
        return True

    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return str(value).strip() == ""


def instance_from_table(rows, n_machines=3, name=None):
    """
    Build a simple instance from a table of jobs.

    ``rows`` may be a pandas DataFrame or a list of dictionaries. It
    must contain an arrival and a burst column; priority and job id are
    optional. Column names are matched case-insensitively and accept
    common aliases (arrival_time, processing time, ...).

    Blank rows are ignored. Job labels are assigned in row order.
    """

    if hasattr(rows, "to_dict"):
        records = rows.to_dict("records")
    else:
        records = list(rows)

    if not records:
        raise ValueError("The input table is empty.")

    normalised = [
        {_normalise_column(key): value for key, value in record.items()}
        for record in records
    ]

    # A table with an "arrival" header but no "burst" header would
    # otherwise fail row-by-row with a confusing message.
    header_fields = {
        field
        for field, aliases in SIMPLE_COLUMN_ALIASES.items()
        if any(alias in normalised[0] for alias in aliases)
    }

    for required in ("arrival", "burst"):
        if required not in header_fields:
            raise ValueError(
                f"Missing required column: {required}. "
                "Use Arrival Time and Burst Time."
            )

    arrivals = []
    bursts = []
    priorities = []

    for row_number, record in enumerate(normalised, start=2):

        arrival = _pick_column(record, "arrival")
        burst = _pick_column(record, "burst")
        priority = _pick_column(record, "priority")

        if _is_blank(arrival) and _is_blank(burst):
            continue

        try:
            arrival = float(arrival)
        except (TypeError, ValueError):
            raise ValueError(
                f"Row {row_number}: arrival time must be a number "
                f"(got {arrival!r})."
            )

        try:
            burst = float(burst)
        except (TypeError, ValueError):
            raise ValueError(
                f"Row {row_number}: burst time must be a number "
                f"(got {burst!r})."
            )

        if not np.isfinite(arrival) or arrival < 0:
            raise ValueError(
                f"Row {row_number}: arrival time must be zero or more."
            )

        if not np.isfinite(burst) or burst <= 0:
            raise ValueError(
                f"Row {row_number}: burst time must be greater than zero."
            )

        if _is_blank(priority):
            priority = 1
        else:
            try:
                priority_number = float(priority)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Row {row_number}: priority must be a whole number "
                    f"(got {priority!r})."
                )

            if (
                not np.isfinite(priority_number)
                or priority_number < 1
                or not float(priority_number).is_integer()
            ):
                raise ValueError(
                    f"Row {row_number}: priority must be a whole number "
                    "of at least 1."
                )

            priority = int(priority_number)

        arrivals.append(arrival)
        bursts.append(burst)
        priorities.append(priority)

    if not arrivals:
        raise ValueError("The input table has no job rows.")

    if len(arrivals) > MAX_JOBS:
        raise ValueError(
            f"At most {MAX_JOBS} jobs are supported; "
            f"the table has {len(arrivals)}."
        )

    return generate_simple_instance(
        n_jobs=len(arrivals),
        n_machines=n_machines,
        arrivals=arrivals,
        bursts=bursts,
        priorities=priorities,
        name=name or "User input"
    )


def simple_table_rows(instance):
    """One dashboard row per job for a simple instance."""

    return [
        {
            "Job ID": f"J{job['job_id']}",
            "Arrival Time": job.get("arrival", 0),
            "Burst Time": job["operations"][0]["time"],
            "Priority": job["priority"]
        }
        for job in instance["jobs"]
    ]


def simple_template_rows(n_jobs=8):
    """Small example table shown in the dashboard and template file."""

    example = [
        (1, 0, 7, 2),
        (2, 0, 5, 1),
        (3, 0, 3, 3),
        (4, 1, 6, 1),
        (5, 2, 2, 2),
        (6, 3, 8, 3),
        (7, 4, 4, 1),
        (8, 5, 3, 2),
        (9, 7, 6, 3),
        (10, 8, 2, 1)
    ]

    for index in range(len(example), int(n_jobs)):

        example.append((
            index + 1,
            index * 2,
            2 + index % 4,
            1 + index % 3
        ))

    return [
        {
            "Job ID": f"J{job_id}",
            "Arrival Time": arrival,
            "Burst Time": burst,
            "Priority": priority
        }
        for job_id, arrival, burst, priority in example[:n_jobs]
    ]


# ==========================================
# DASHBOARD PRESETS
# ==========================================

PRESETS = {
    "Small — 10 jobs / 3 machines": dict(
        seed=11, n_jobs=10, n_machines=3
    ),
    "Medium — 15 jobs / 4 machines": dict(
        seed=22, n_jobs=15, n_machines=4
    ),
    "Large — 20 jobs / 5 machines": dict(
        seed=33, n_jobs=20, n_machines=5
    )
}


def preset_instance(name):
    """Return a fixed, reproducible preset instance."""

    config = PRESETS[name]

    return generate_instance(
        seed=config["seed"],
        n_jobs=config["n_jobs"],
        n_machines=config["n_machines"],
        name=name
    )


def instance_to_rows(instance):
    """Flatten an instance into one row per operation (dashboard table)."""

    rows = []

    for job in instance["jobs"]:

        for op_index, operation in enumerate(job["operations"]):

            rows.append({
                "Job": f"J{job['job_id']}",
                "Operation": op_index + 1,
                "Machine": (
                    "Any"
                    if operation["machine"] is None
                    else f"M{operation['machine'] + 1}"
                ),
                "Processing time": operation["time"],
                "Priority": job["priority"],
                "Arrival": job.get("arrival", 0)
            })

    return rows


# ==========================================
# JOB-SHOP ENVIRONMENT
# ==========================================

class JobShopEnv(gym.Env):
    """Gymnasium environment for job-shop dispatching."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_machines=MAX_MACHINES,
        max_jobs=MAX_JOBS,
        reward_mode="increment",
        simple=False
    ):

        super().__init__()

        # "increment": dense horizon increments (default)
        # "terminal": one makespan reward at the end of the episode
        if reward_mode not in ("increment", "terminal"):
            raise ValueError(
                f"Unknown reward_mode: {reward_mode}"
            )

        self.reward_mode = reward_mode
        self.simple = bool(simple)

        self.max_machines = int(max_machines)
        self.max_jobs = int(max_jobs)
        self.n_actions = self.max_jobs

        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32
        )

        self.instance = None
        self.n_machines = 0
        self.n_jobs = 0

        self.machine_free = np.zeros(self.max_machines)
        self.busy = np.zeros(self.max_machines)

        self.next_op = np.zeros(self.max_jobs, dtype=int)
        self.job_ready = np.zeros(self.max_jobs)
        self.job_completion = np.full(self.max_jobs, np.nan)
        self.job_total_proc = np.zeros(self.max_jobs)
        self.job_arrival = np.zeros(self.max_jobs)

        self.total_work = 0.0
        self.total_ops = 0
        self.completed_ops = 0

        self.time_scale = 1.0
        self.lower_bound = 1.0
        self.processing_scale = float(MAX_PROCESSING_TIME)
        self.priority_scale = 3.0
        self.decision_time = 0.0

        self.schedule = []
        self.current_mask = np.zeros(self.max_jobs, dtype=bool)

    # ==========================================
    # RESET
    # ==========================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        options = options or {}

        if "instance" in options:
            instance = options["instance"]
        elif self.simple:
            # Training instances use a reserved seed range so they can
            # never overlap the fixed evaluation seeds (10,000-10,099).
            instance = generate_simple_instance(
                seed=1_000_000 + int(self.np_random.integers(0, 1_000_000)),
                n_jobs=int(self.np_random.integers(5, MAX_JOBS + 1)),
                n_machines=int(self.np_random.integers(1, MAX_MACHINES + 1))
            )
        else:
            # Training instances use a reserved seed range so they can
            # never overlap the fixed evaluation seeds (10,000-10,099).
            instance = generate_instance(
                seed=1_000_000 + int(self.np_random.integers(0, 1_000_000))
            )

        self.instance = instance
        self.n_machines = int(instance["n_machines"])
        self.n_jobs = int(instance["n_jobs"])

        self.machine_free = np.zeros(self.max_machines)
        self.busy = np.zeros(self.max_machines)

        self.next_op = np.zeros(self.max_jobs, dtype=int)
        self.job_ready = np.zeros(self.max_jobs)
        self.job_completion = np.full(self.max_jobs, np.nan)
        self.job_total_proc = np.zeros(self.max_jobs)
        self.job_arrival = np.zeros(self.max_jobs)

        for job_index, job in enumerate(instance["jobs"]):

            self.job_total_proc[job_index] = sum(
                operation["time"] for operation in job["operations"]
            )

            self.job_arrival[job_index] = float(job.get("arrival", 0))
            self.job_ready[job_index] = self.job_arrival[job_index]

        max_op_time = max(
            (
                float(operation["time"])
                for job in instance["jobs"]
                for operation in job["operations"]
            ),
            default=float(MAX_PROCESSING_TIME)
        )
        self.processing_scale = max(float(MAX_PROCESSING_TIME), max_op_time)
        self.priority_scale = max(
            3.0,
            float(max(job.get("priority", 1) for job in instance["jobs"]))
        )

        self.total_work = float(self.job_total_proc.sum())
        self.total_ops = int(sum(
            len(job["operations"]) for job in instance["jobs"]
        ))
        self.completed_ops = 0

        longest_job = float(self.job_total_proc.max()) if self.n_jobs else 0.0
        latest_arrival = (
            float(self.job_arrival.max()) if self.n_jobs else 0.0
        )
        longest_release = max(
            (
                self.job_arrival[index] + self.job_total_proc[index]
                for index in range(self.n_jobs)
            ),
            default=0.0
        )

        self.lower_bound = max(
            latest_arrival + self.total_work / max(self.n_machines, 1),
            longest_release,
            1.0
        )
        self.time_scale = self.lower_bound

        self.decision_time = 0.0
        self.schedule = []

        self.current_mask = self._compute_mask()

        return self.get_observation(), {}

    # ==========================================
    # ELIGIBLE ACTIONS
    # ==========================================

    def _pending_starts(self):
        """Earliest possible start time of every job's next operation."""

        starts = np.full(self.max_jobs, np.inf)

        for job_index in range(self.n_jobs):

            op_index = int(self.next_op[job_index])
            operations = self.instance["jobs"][job_index]["operations"]

            if op_index >= len(operations):
                continue

            machine = operations[op_index].get("machine")

            if machine is None:
                machine_free = float(
                    self.machine_free[:self.n_machines].min()
                )
            else:
                machine_free = float(self.machine_free[machine])

            starts[job_index] = max(
                machine_free,
                self.job_ready[job_index]
            )

        return starts

    def _compute_mask(self):
        """
        A job is eligible when its next operation can start at the
        earliest currently achievable time. This is the "next operation
        can currently be scheduled" rule from the project document.
        """

        mask = np.zeros(self.max_jobs, dtype=bool)
        starts = self._pending_starts()
        earliest = starts.min()

        if not np.isfinite(earliest):
            self.decision_time = float(self.machine_free.max())
            return mask

        self.decision_time = float(earliest)

        mask = starts <= earliest + 1e-9

        return mask

    def action_masks(self):
        """Boolean mask of jobs the agent is allowed to choose."""

        self.current_mask = self._compute_mask()
        return self.current_mask.copy()

    # ==========================================
    # OBSERVATION
    # ==========================================

    def get_observation(self):

        mask = self._compute_mask()
        self.current_mask = mask

        now = self.decision_time
        safe_now = max(now, 1e-6)
        scale = max(self.time_scale, 1e-6)

        # MACHINE FEATURES
        machine_features = np.zeros(
            (self.max_machines, MACHINE_FEATURES), dtype=np.float32
        )

        pending_by_machine = np.zeros(self.max_machines)
        workload_by_machine = np.zeros(self.max_machines)

        for job_index in range(self.n_jobs):

            op_index = int(self.next_op[job_index])
            operations = self.instance["jobs"][job_index]["operations"]

            if op_index >= len(operations):
                continue

            machine = operations[op_index]["machine"]

            if machine is None:
                # Flexible operation: attribute it to the machine that
                # would run it right now, matching the scheduling step.
                machine = int(
                    np.argmin(self.machine_free[:self.n_machines])
                )

            pending_by_machine[machine] += 1
            workload_by_machine[machine] += operations[op_index]["time"]

        for machine in range(self.n_machines):

            machine_features[machine] = [
                1.0,
                np.clip(
                    (self.machine_free[machine] - now) / scale,
                    0.0, 1.0
                ),
                np.clip(self.busy[machine] / safe_now, 0.0, 1.0),
                np.clip(
                    pending_by_machine[machine] / max(self.n_jobs, 1),
                    0.0, 1.0
                ),
                np.clip(
                    workload_by_machine[machine] / max(self.total_work, 1e-6),
                    0.0, 1.0
                )
            ]

        # JOB FEATURES
        job_features = np.zeros(
            (self.max_jobs, JOB_FEATURES), dtype=np.float32
        )

        for job_index in range(self.n_jobs):

            job = self.instance["jobs"][job_index]
            op_index = int(self.next_op[job_index])
            operations = job["operations"]
            completed = op_index >= len(operations)

            machine_onehot = np.zeros(self.max_machines, dtype=np.float32)
            processing_time = 0.0
            remaining_ops = 0.0
            remaining_work = 0.0

            if not completed:

                operation = operations[op_index]

                if operation.get("machine") is None:
                    machine_onehot[
                        int(np.argmin(self.machine_free[:self.n_machines]))
                    ] = 1.0
                else:
                    machine_onehot[operation["machine"]] = 1.0

                processing_time = operation["time"]

                remaining = operations[op_index:]

                remaining_ops = len(remaining) / MAX_OPERATIONS
                remaining_work = (
                    sum(op["time"] for op in remaining)
                    / max(self.total_work, 1e-6)
                )

            ready = (
                not completed
                and self.job_ready[job_index] <= now + 1e-9
            )

            waiting = (
                max(0.0, now - self.job_ready[job_index]) / scale
                if not completed else 0.0
            )

            job_features[job_index] = np.concatenate([
                np.array([
                    1.0,
                    1.0 if completed else 0.0
                ], dtype=np.float32),
                machine_onehot,
                np.array([
                    processing_time / self.processing_scale,
                    remaining_ops,
                    remaining_work,
                    min(waiting, 1.0),
                    job["priority"] / self.priority_scale,
                    1.0 if ready else 0.0
                ], dtype=np.float32)
            ])

        # GLOBAL FEATURES
        global_features = np.array([
            np.clip(now / scale, 0.0, 1.0),
            self.completed_ops / max(self.total_ops, 1),
            self.n_jobs / MAX_JOBS,
            self.n_machines / MAX_MACHINES
        ], dtype=np.float32)

        # ACTION MASK (kept inside the observation for the masked DQN)
        mask_features = mask.astype(np.float32)

        observation = np.concatenate([
            machine_features.flatten(),
            job_features.flatten(),
            global_features,
            mask_features
        ])

        return np.clip(observation, 0.0, 1.0).astype(np.float32)

    # ==========================================
    # STEP
    # ==========================================

    def step(self, action):

        action = int(action)

        mask = self._compute_mask()
        self.current_mask = mask

        invalid = not bool(mask[action])

        if invalid:

            valid_actions = np.flatnonzero(mask)

            if len(valid_actions) == 0:
                # Episode is already finished.
                return (
                    self.get_observation(),
                    0.0,
                    True,
                    False,
                    {"invalid_action": True}
                )

            action = int(valid_actions[0])

        job_index = action
        job = self.instance["jobs"][job_index]
        operations = job["operations"]

        # Makespan before this operation, used by the dense reward below.
        previous_horizon = float(self.machine_free.max())

        op_index = int(self.next_op[job_index])
        operation = operations[op_index]

        machine = operation.get("machine")

        if machine is None:
            # Flexible operation: use the machine that is free earliest.
            machine = int(
                np.argmin(self.machine_free[:self.n_machines])
            )

        machine = int(machine)
        processing_time = float(operation["time"])

        start = max(
            self.machine_free[machine],
            self.job_ready[job_index]
        )
        waiting = start - self.job_ready[job_index]
        idle = start - self.machine_free[machine]
        end = start + processing_time

        # UPDATE MACHINE AND JOB STATE
        self.machine_free[machine] = end
        self.job_ready[job_index] = end
        self.busy[machine] += processing_time

        self.next_op[job_index] += 1
        self.completed_ops += 1

        if self.next_op[job_index] >= len(operations):
            self.job_completion[job_index] = end

        # RECORD THE OPERATION
        self.schedule.append({
            "job_id": int(job["job_id"]),
            "operation": op_index + 1,
            "machine": int(machine),
            "processing_time": float(processing_time),
            "start": float(start),
            "end": float(end),
            "waiting_time": float(waiting),
            "idle_time": float(idle),
            "priority": int(job["priority"])
        })

        # REWARD
        #
        # "increment" mode uses the dense horizon increment, which
        # telescopes to the final makespan:
        #
        #   sum(new_horizon - previous_horizon) = final makespan
        #
        # "terminal" mode puts the whole makespan signal at the end,
        # which avoids the local bias that makes dense horizon
        # increments prefer short jobs.
        #
        # Rewards are multiplied by REWARD_SCALE so the Q-value
        # differences are large enough for stable DQN updates. A
        # positive constant scale does not change the optimal policy.
        terminated = self.completed_ops >= self.total_ops

        if self.reward_mode == "terminal":

            reward = 0.0

            if terminated:

                makespan = float(self.machine_free.max())
                total_waiting = sum(
                    entry["waiting_time"] for entry in self.schedule
                )

                reward = -makespan / self.time_scale
                reward -= 0.005 * total_waiting / self.time_scale
                reward += 0.1

        else:

            new_horizon = float(self.machine_free.max())
            reward = -(new_horizon - previous_horizon) / self.time_scale

            # Small secondary pressure towards shorter job waiting times.
            # Kept low so that makespan stays the dominant objective.
            reward -= 0.005 * waiting / self.time_scale

            if terminated:
                # Small fixed completion bonus; the makespan itself is
                # already accounted for by the telescoping increment.
                reward += 0.1

        if invalid:
            reward -= 1.0

        reward *= REWARD_SCALE

        observation = self.get_observation()

        info = {
            "job_id": int(job["job_id"]),
            "machine": int(machine),
            "operation": op_index + 1,
            "processing_time": processing_time,
            "start": float(start),
            "end": float(end),
            "waiting_time": float(waiting),
            "invalid_action": invalid,
            "completed_ops": self.completed_ops,
            "total_ops": self.total_ops
        }

        return observation, float(reward), terminated, False, info

    # ==========================================
    # METRICS AND SUMMARY
    # ==========================================

    def metrics(self):

        makespan = float(self.machine_free.max()) if self.n_machines else 0.0

        flow_times = self.job_completion - self.job_arrival
        waiting_times = flow_times - self.job_total_proc

        mean_waiting = float(np.nanmean(waiting_times)) if self.n_jobs else 0.0
        total_waiting = float(np.nansum(waiting_times)) if self.n_jobs else 0.0

        utilization = (
            self.busy[:self.n_machines] / max(makespan, 1e-9)
        )

        throughput = self.n_jobs / makespan if makespan > 0 else 0.0

        return {
            "makespan": makespan,
            "mean_waiting_time": mean_waiting,
            "total_waiting_time": total_waiting,
            "throughput": throughput,
            "throughput_per_100": throughput * 100.0,
            "mean_utilization": float(np.mean(utilization)) * 100.0,
            "machine_utilization": [
                float(value) * 100.0 for value in utilization
            ],
            "lower_bound": float(self.lower_bound),
            "n_jobs": int(self.n_jobs),
            "n_machines": int(self.n_machines),
            "total_operations": int(self.total_ops)
        }

    def summary(self):

        result = self.metrics()
        result["schedule"] = list(self.schedule)
        result["instance"] = self.instance

        return result

    # ==========================================
    # RENDER
    # ==========================================

    def render(self):

        result = self.metrics()

        print("\n========== JOB-SHOP SCHEDULE ==========")
        print("Jobs:", result["n_jobs"])
        print("Machines:", result["n_machines"])
        print("Makespan:", round(result["makespan"], 2))
        print("Mean waiting time:", round(result["mean_waiting_time"], 2))
        print("Throughput (/100 time):", round(result["throughput_per_100"], 2))
        print("Mean utilization (%):", round(result["mean_utilization"], 2))
        print("=======================================")


# ==========================================
# BASELINE DISPATCH RULES
# ==========================================

def _choose_dispatch_action(env, strategy, rng):
    """Pick an eligible job with a fixed dispatch rule."""

    valid = np.flatnonzero(env.action_masks())

    if len(valid) == 0:
        return 0

    if strategy == "random":
        return int(rng.choice(valid))

    if strategy == "spt":

        def key(job_index):
            op_index = int(env.next_op[job_index])
            operation = env.instance["jobs"][job_index]["operations"][op_index]
            return (operation["time"], job_index)

        return int(min(valid, key=key))

    if strategy == "fcfs":

        def key(job_index):
            return (
                env.instance["jobs"][job_index].get("arrival", 0),
                job_index
            )

        return int(min(valid, key=key))

    if strategy == "sjf":

        def key(job_index):
            op_index = int(env.next_op[job_index])
            operations = env.instance["jobs"][job_index]["operations"]
            remaining = sum(
                operation["time"] for operation in operations[op_index:]
            )
            return (remaining, job_index)

        return int(min(valid, key=key))

    if strategy == "priority":

        def key(job_index):
            job = env.instance["jobs"][job_index]
            return (
                job.get("priority", 1),
                env.job_arrival[job_index],
                job_index
            )

        return int(min(valid, key=key))

    raise ValueError(f"Unknown dispatch strategy: {strategy}")


def run_dispatch(instance, strategy="fcfs", seed=0, env=None):
    """
    Run one dispatch rule on an instance.

    Valid strategies: random, fcfs, spt, sjf and priority. The same
    instance object is reused by every algorithm in the evaluation, so
    all comparisons are paired.
    """

    if strategy not in DISPATCH_RULES:
        raise ValueError(f"Unknown dispatch strategy: {strategy}")

    env = env or JobShopEnv()

    env.reset(seed=seed, options={"instance": instance})

    rng = np.random.default_rng(seed)

    done = False

    while not done:

        action = _choose_dispatch_action(env, strategy, rng)
        _, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

    return env


# ==========================================
# TRAINED AGENT
# ==========================================

def _predict_action(model, observation, mask):
    """Predict one action and enforce the environment mask as a safety net."""

    try:
        action, _ = model.predict(
            observation, deterministic=True, action_masks=mask
        )
    except TypeError:
        # The custom MaskedDQN reads the mask from the observation.
        action, _ = model.predict(observation, deterministic=True)

    action = int(np.asarray(action).reshape(-1)[0])

    if not bool(mask[action]):
        valid = np.flatnonzero(mask)
        action = int(valid[0]) if len(valid) else 0

    return action


def run_agent(model, instance, env=None):
    """Run a trained DQN or PPO policy on one instance."""

    env = env or JobShopEnv()

    observation, _ = env.reset(seed=0, options={"instance": instance})

    done = False

    while not done:

        mask = env.action_masks()
        action = _predict_action(model, observation, mask)

        observation, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return env
