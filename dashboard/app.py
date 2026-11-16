
"""
Production Optimization using RL: scheduling dashboard.

There are two pages:

1. **Your data**: enter jobs by hand or upload Excel/CSV (arrival time,
   burst time, priority), choose the number of identical machines and
   compare FCFS, SJF, Priority, DQN and PPO on the same data.
2. **Job-shop demo**: the original job-shop study, where every job visits
   several machines. It is here so the job-shop DQN and PPO results stay
   reproducible from the dashboard.

The simple page uses the arrival + burst + priority model: every job has
one operation that can run on any machine.
"""

import sys

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as web
import altair as alt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from environment.job_shop_env import (
    PRESETS, generate_instance, instance_from_table, instance_to_rows,
    preset_instance, run_agent, run_dispatch, simple_table_rows,
    simple_template_rows
)


SIMPLE_DQN_PATH = BASE_DIR / "models" / "simple_dqn.zip"
SIMPLE_PPO_PATH = BASE_DIR / "models" / "simple_ppo.zip"
JOBSHOP_DQN_PATH = BASE_DIR / "models" / "jobshop_dqn.zip"
JOBSHOP_PPO_PATH = BASE_DIR / "models" / "jobshop_ppo.zip"

DISPATCH_NAMES = {
    "Random": "random",
    "FCFS": "fcfs",
    "SJF": "sjf",
    "SPT": "spt",
    "Priority": "priority"
}

SIMPLE_ORDER = ("Random", "FCFS", "SJF", "Priority", "DQN", "PPO")
JOBSHOP_ORDER = ("Random", "FCFS", "SPT", "SJF", "Priority", "DQN", "PPO")

LOWER_IS_BETTER = {
    "makespan": True,
    "mean_waiting_time": True,
    "throughput_per_100": False,
    "mean_utilization": False
}


# ==========================================
# MODEL LOADING
# ==========================================

@web.cache_resource(show_spinner=False)
def load_simple_models():

    models = {}

    if SIMPLE_DQN_PATH.exists():

        from agents.masked_dqn import MaskedDQN

        models["DQN"] = MaskedDQN.load(str(SIMPLE_DQN_PATH))

    if SIMPLE_PPO_PATH.exists():

        from sb3_contrib import MaskablePPO

        models["PPO"] = MaskablePPO.load(str(SIMPLE_PPO_PATH))

    return models


@web.cache_resource(show_spinner=False)
def load_jobshop_models():

    models = {}

    if JOBSHOP_DQN_PATH.exists():

        from agents.masked_dqn import MaskedDQN

        models["DQN"] = MaskedDQN.load(str(JOBSHOP_DQN_PATH))

    if JOBSHOP_PPO_PATH.exists():

        from sb3_contrib import MaskablePPO

        models["PPO"] = MaskablePPO.load(str(JOBSHOP_PPO_PATH))

    return models


# ==========================================
# RUN ALGORITHMS
# ==========================================

def evaluate_instance(instance, algorithm_names, models):

    results = {}
    errors = {}

    for name in algorithm_names:

        try:

            if name in models:

                env = run_agent(models[name], instance)

            else:

                env = run_dispatch(
                    instance,
                    strategy=DISPATCH_NAMES[name],
                    seed=0
                )

            results[name] = env.summary()

        except Exception as error:  # keep the other algorithms running

            errors[name] = f"{type(error).__name__}: {error}"

    return results, errors


def best_rule(results, key, baselines):

    """Best dispatch-rule value for one metric, and the rule name."""

    candidates = [
        (name, results[name][key])
        for name in baselines
        if name in results
    ]

    if not candidates:
        return None, 0.0

    if LOWER_IS_BETTER[key]:
        name, value = min(candidates, key=lambda item: item[1])
    else:
        name, value = max(candidates, key=lambda item: item[1])

    return name, value


# ==========================================
# CHARTS
# ==========================================

def gantt_chart(schedule, n_machines, simple=False):

    frame = pd.DataFrame(schedule)

    frame["Machine"] = frame["machine"].map(lambda m: f"M{m + 1}")
    frame["Job"] = frame["job_id"].map(lambda j: f"J{j}")

    if simple:
        frame["Label"] = frame["Job"]
    else:
        frame["Label"] = frame["Job"] + "·" + frame["operation"].astype(str)

    machine_order = [f"M{m + 1}" for m in range(n_machines)]

    base = alt.Chart(frame).encode(
        x=alt.X("start:Q", title="Time"),
        x2="end:Q",
        y=alt.Y("Machine:N", title=None, sort=machine_order)
    )

    bars = base.mark_bar(
        cornerRadius=2,
        stroke="white",
        strokeWidth=0.6
    ).encode(
        color=alt.Color(
            "Job:N",
            legend=None,
            scale=alt.Scale(scheme="tableau20")
        ),
        tooltip=[
            alt.Tooltip("Job:N", title="Job"),
            alt.Tooltip("Machine:N", title="Machine"),
            alt.Tooltip("operation:Q", title="Operation"),
            alt.Tooltip("start:Q", title="Start"),
            alt.Tooltip("end:Q", title="End"),
            alt.Tooltip("processing_time:Q", title="Processing time"),
            alt.Tooltip("waiting_time:Q", title="Waiting time")
        ]
    )

    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=2,
        fontSize=10
    ).encode(
        text=alt.Text("Label:N"),
        color=alt.value("#1f2937")
    )

    return (bars + labels).properties(
        height=max(190, 62 * n_machines)
    )


def comparison_chart(results, order, focus):

    rows = []

    for algorithm in order:

        if algorithm not in results:
            continue

        rows.append({
            "Algorithm": algorithm,
            "Makespan": results[algorithm]["makespan"],
            "Focus": algorithm == focus
        })

    frame = pd.DataFrame(rows)

    chart = alt.Chart(frame).mark_bar(
        cornerRadius=4
    ).encode(
        x=alt.X(
            "Algorithm:N",
            sort=[name for name in order if name in results],
            title=None
        ),
        y=alt.Y("Makespan:Q", title="Makespan"),
        color=alt.condition(
            alt.datum.Focus,
            alt.value("#2563eb"),
            alt.value("#cbd5e1")
        ),
        tooltip=[
            alt.Tooltip("Algorithm:N"),
            alt.Tooltip("Makespan:Q", format=".1f")
        ]
    )

    text = chart.mark_text(
        dy=-6,
        fontSize=11
    ).encode(
        text=alt.Text("Makespan:Q", format=".1f")
    )

    return (chart + text).properties(height=240)


# ==========================================
# SHARED RESULTS VIEW
# ==========================================

def render_results(instance, results, order, baselines, simple, key_prefix):

    available = [name for name in order if name in results]

    if not available:
        web.warning("No algorithm produced a schedule for this input.")
        return

    default_focus = (
        "DQN" if "DQN" in available
        else "PPO" if "PPO" in available
        else available[0]
    )

    focus = web.radio(
        "Show schedule for",
        available,
        index=available.index(default_focus),
        horizontal=True,
        key=f"{key_prefix}_focus_" + "-".join(available)
    )

    focused = results[focus]
    best_makespan_name, best_makespan = best_rule(
        results, "makespan", baselines
    )

    web.subheader("Production KPIs")

    columns = web.columns(4)

    kpi_specs = [
        ("Makespan", "makespan", "{:.1f}", "lower"),
        ("Mean waiting time", "mean_waiting_time", "{:.1f}", "lower"),
        ("Throughput / 100 time", "throughput_per_100", "{:.2f}", "higher"),
        ("Machine utilization", "mean_utilization", "{:.1f}%", "higher")
    ]

    for column, (label, key, fmt, direction) in zip(columns, kpi_specs):

        value = focused[key]
        reference_name, reference = best_rule(results, key, baselines)

        if reference_name is None:
            delta_text = "no rule baseline"
            color = "off"
            caption_text = "Select a dispatch rule to compare."
        elif abs(reference) > 1e-9:
            delta = (value - reference) / reference * 100.0
            if focus == reference_name:
                delta_text = "best rule baseline"
                color = "off"
            else:
                delta_text = f"{delta:+.1f}% vs {reference_name}"
                color = "inverse" if direction == "lower" else "normal"
            caption_text = (
                f"Best rule ({reference_name}): {fmt.format(reference)}"
            )
        else:
            delta_text = f"vs {reference_name}"
            color = "off"
            caption_text = f"Best rule ({reference_name}): {fmt.format(reference)}"

        with column:

            web.metric(
                label,
                fmt.format(value),
                delta=delta_text,
                delta_color=color
            )
            web.caption(caption_text)

    web.subheader(f"Schedule: {focus}")

    if simple:
        web.caption(
            "Each bar is one job. The label is the job ID; hover for "
            "arrival, start and end times."
        )
    else:
        web.caption(
            "Each bar is one operation. The label shows the job and its "
            "operation number (J3·2 = job 3, second operation)."
        )

    web.altair_chart(
        gantt_chart(focused["schedule"], instance["n_machines"], simple),
        width="stretch"
    )

    if simple:

        schedule_frame = pd.DataFrame(focused["schedule"])[
            [
                "job_id", "machine", "start", "end",
                "processing_time", "waiting_time"
            ]
        ].rename(columns={
            "job_id": "Job",
            "machine": "Machine",
            "start": "Start",
            "end": "End",
            "processing_time": "Burst",
            "waiting_time": "Waiting"
        })
        schedule_frame["Job"] = schedule_frame["Job"].map(lambda j: f"J{j}")
        schedule_frame["Machine"] = schedule_frame["Machine"].map(
            lambda m: f"M{m + 1}"
        )

    else:

        schedule_frame = pd.DataFrame(focused["schedule"])[
            [
                "job_id", "operation", "machine", "start", "end",
                "processing_time", "waiting_time"
            ]
        ].rename(columns={
            "job_id": "Job",
            "operation": "Operation",
            "machine": "Machine",
            "start": "Start",
            "end": "End",
            "processing_time": "Processing time",
            "waiting_time": "Waiting"
        })
        schedule_frame["Job"] = schedule_frame["Job"].map(lambda j: f"J{j}")
        schedule_frame["Machine"] = schedule_frame["Machine"].map(
            lambda m: f"M{m + 1}"
        )

    with web.expander("Schedule table"):

        web.dataframe(
            schedule_frame,
            hide_index=True,
            width="stretch"
        )

        web.download_button(
            "Download schedule CSV",
            schedule_frame.to_csv(index=False),
            file_name=f"schedule_{key_prefix}_{focus}.csv",
            mime="text/csv",
            key=f"{key_prefix}_download"
        )

    web.subheader("Algorithm comparison")

    table_rows = []

    for algorithm in available:

        result = results[algorithm]

        if best_makespan_name is not None and abs(best_makespan) > 1e-9:
            makespan_delta = (
                (result["makespan"] - best_makespan)
                / best_makespan * 100.0
            )
            makespan_text = f"{makespan_delta:+.1f}%"
        else:
            makespan_text = "—"

        table_rows.append({
            "Algorithm": algorithm,
            "Makespan": round(result["makespan"], 1),
            "Mean waiting": round(result["mean_waiting_time"], 1),
            "Throughput / 100": round(result["throughput_per_100"], 2),
            "Utilization %": round(result["mean_utilization"], 1),
            "Makespan vs best rule": makespan_text
        })

    web.dataframe(
        pd.DataFrame(table_rows),
        hide_index=True,
        width="stretch"
    )

    web.altair_chart(
        comparison_chart(results, order, focus),
        width="stretch"
    )

    if best_makespan_name is not None:
        web.caption(
            f"Best dispatch rule on this instance: **{best_makespan_name}** "
            f"({best_makespan:.1f} makespan). All algorithms ran on the "
            "same jobs and machines."
        )


# ==========================================
# PAGE 1 — USER INPUT
# ==========================================

def render_user_page():

    web.header("Schedule your own jobs")

    web.markdown(
        "Every job has an **arrival time**, a **burst time** and a "
        "**priority**. Each job runs once on any free machine. Use "
        "manual entry for a few jobs, or upload a spreadsheet."
    )

    control_left, control_right = web.columns([2, 1])

    with control_left:

        input_mode = web.radio(
            "Input method",
            ("Manual entry", "Excel upload"),
            horizontal=True
        )

    with control_right:

        n_machines = int(web.number_input(
            "Number of machines",
            min_value=1,
            max_value=5,
            value=3,
            step=1
        ))

    instance = None
    input_label = ""

    if input_mode == "Manual entry":

        n_jobs = int(web.number_input(
            "Number of jobs",
            min_value=1,
            max_value=20,
            value=8,
            step=1
        ))

        table_key = f"manual_table_{n_jobs}"

        if table_key not in web.session_state:

            web.session_state[table_key] = pd.DataFrame(
                simple_template_rows(n_jobs)
            )

        edited = web.data_editor(
            web.session_state[table_key],
            num_rows="fixed",
            hide_index=True,
            width="stretch",
            key=f"manual_editor_{n_jobs}",
            column_config={
                "Job ID": web.column_config.TextColumn(
                    "Job ID",
                    help="Label from your sheet; schedules use J1…Jn."
                ),
                "Arrival Time": web.column_config.NumberColumn(
                    "Arrival Time", min_value=0.0, step=1.0, format="%.0f"
                ),
                "Burst Time": web.column_config.NumberColumn(
                    "Burst Time", min_value=1.0, step=1.0, format="%.0f"
                ),
                "Priority": web.column_config.NumberColumn(
                    "Priority", min_value=1, max_value=5, step=1,
                    format="%d"
                )
            }
        )

        try:
            instance = instance_from_table(
                edited, n_machines=n_machines, name="Manual entry"
            )
            input_label = f"{instance['n_jobs']} manually entered jobs"

        except ValueError as error:
            web.error(str(error))

    else:

        uploaded = web.file_uploader(
            "Upload jobs (.xlsx, .xlsm or .csv)",
            type=["xlsx", "xlsm", "csv"],
            help=(
                "Required columns: Arrival Time and Burst Time. "
                "Optional: Job ID and Priority. Column names are "
                "case-insensitive."
            )
        )

        if uploaded is None:

            web.info(
                "Upload a spreadsheet, or download the template from "
                "the sidebar to see the expected layout."
            )

        else:

            try:

                if uploaded.name.lower().endswith(".csv"):
                    raw_frame = pd.read_csv(uploaded)
                else:
                    raw_frame = pd.read_excel(uploaded)

                instance = instance_from_table(
                    raw_frame,
                    n_machines=n_machines,
                    name=uploaded.name
                )
                input_label = f"{instance['n_jobs']} jobs from {uploaded.name}"

            except Exception as error:

                web.error(f"Could not read the file: {error}")

            else:

                with web.expander("Uploaded data", expanded=False):
                    web.dataframe(
                        raw_frame,
                        hide_index=True,
                        width="stretch"
                    )

    web.subheader("Schedulers")

    simple_models = load_simple_models()
    missing_simple = [
        name for name in ("DQN", "PPO") if name not in simple_models
    ]

    default_schedulers = [
        name for name in SIMPLE_ORDER if name not in missing_simple
    ]

    selected = web.multiselect(
        "Algorithms to compare",
        [name for name in SIMPLE_ORDER],
        default=default_schedulers,
        help=(
            "SJF = shortest job first (same as shortest processing time "
            "when every job has one operation). Priority uses 1 as the "
            "highest priority."
        )
    )

    if missing_simple:
        web.caption(
            "Unavailable (model file missing): "
            + ", ".join(missing_simple)
            + ". The rules still work. Train with "
            "`venv\\Scripts\\python.exe agents\\train_dqn_simple.py` "
            "and `agents\\train_ppo_simple.py`."
        )

    if instance is None:

        web.info("Enter valid jobs above to run the schedulers.")

    elif not selected:

        web.warning("Select at least one scheduler.")

    else:

        with web.spinner("Running the schedulers..."):

            results, errors = evaluate_instance(
                instance,
                [name for name in selected if name in SIMPLE_ORDER],
                simple_models
            )

        web.success(
            f"Instance ready: {input_label} · {n_machines} machines"
        )

        for name, message in errors.items():
            web.error(f"{name}: {message}")

        baselines = [
            name for name in ("Random", "FCFS", "SJF", "Priority")
            if name in results
        ]

        render_results(
            instance,
            results,
            [name for name in SIMPLE_ORDER if name in results],
            baselines,
            simple=True,
            key_prefix="simple"
        )

        with web.expander("Input jobs"):
            web.dataframe(
                pd.DataFrame(simple_table_rows(instance)),
                hide_index=True,
                width="stretch"
            )

        web.info(
            "The DQN and PPO agents were trained on random instances "
            "with 5–20 jobs, 1–5 machines, arrivals 0–10, burst times "
            "1–9 and priorities 1–3. The rules are exact and work for "
            "any values."
        )


# ==========================================
# PAGE 2 — JOB-SHOP DEMO
# ==========================================

def render_jobshop_page():

    web.header("Job-shop demo")

    web.markdown(
        "This is the original job-shop problem. Every job visits 2–4 "
        "machines in a fixed order, and a job can only run when its next "
        "operation can start. The page is here so you can reproduce the "
        "job-shop DQN and PPO results."
    )

    preset_name = web.selectbox(
        "Production instance",
        list(PRESETS) + ["Random instance"],
        key="jobshop_preset"
    )

    if preset_name == "Random instance":

        random_columns = web.columns(3)

        with random_columns[0]:

            seed = web.number_input(
                "Instance seed",
                min_value=0,
                max_value=999_999,
                value=42,
                step=1,
                key="jobshop_seed"
            )

        with random_columns[1]:

            n_jobs_shop = web.slider(
                "Number of jobs",
                min_value=10,
                max_value=20,
                value=12,
                key="jobshop_jobs"
            )

        with random_columns[2]:

            n_machines_shop = web.slider(
                "Number of machines",
                min_value=3,
                max_value=5,
                value=4,
                key="jobshop_machines"
            )

        instance_shop = generate_instance(
            seed=int(seed),
            n_jobs=int(n_jobs_shop),
            n_machines=int(n_machines_shop),
            name=f"Random (seed {int(seed)})"
        )

    else:

        instance_shop = preset_instance(preset_name)

    web.caption(
        f"{instance_shop['n_jobs']} jobs · {instance_shop['n_machines']} "
        f"machines · "
        f"{sum(len(j['operations']) for j in instance_shop['jobs'])} "
        "operations"
    )

    jobshop_models = load_jobshop_models()

    shop_order = [
        name for name in JOBSHOP_ORDER
        if name not in ("DQN", "PPO") or name in jobshop_models
    ]

    shop_selected = web.multiselect(
        "Algorithms to compare",
        shop_order,
        default=shop_order,
        key="jobshop_algorithms"
    )

    if not shop_selected:

        web.warning("Select at least one scheduler.")

    else:

        with web.spinner("Running the job-shop schedulers..."):

            shop_results, shop_errors = evaluate_instance(
                instance_shop,
                shop_selected,
                jobshop_models
            )

        for name, message in shop_errors.items():
            web.error(f"{name}: {message}")

        render_results(
            instance_shop,
            shop_results,
            [name for name in shop_order if name in shop_results],
            [
                name for name in
                ("Random", "FCFS", "SPT", "SJF", "Priority")
                if name in shop_results
            ],
            simple=False,
            key_prefix="jobshop"
        )

        with web.expander("Instance details (jobs and operations)"):
            web.dataframe(
                pd.DataFrame(instance_to_rows(instance_shop)),
                hide_index=True,
                width="stretch"
            )

        with web.expander("How the job-shop scheduler works"):

            web.markdown(
                """
**State:** machine availability, utilization and queue length, plus
each job's next machine, processing time, waiting time, priority and
remaining operations.

**Action:** choose one *eligible* job, meaning a job whose next operation
can start now on a free machine. Ineligible jobs are masked out.

**Reward:** the schedule is scored at the end of the episode. It is the
negative makespan plus a small penalty for total waiting time, normalised
by the instance's lower bound.

**Training:** MWKR warm start plus 400,000 DQN fine-tuning steps for DQN,
and 200,000 steps for PPO. Evaluation uses 100 unseen instances that are
identical for every algorithm. Full results are in
`reports/jobshop_comparison.csv`.
                """
            )

        with web.expander("Model details"):

            web.markdown(
                """
- **DQN (primary):** `models/jobshop_dqn.zip`, masked Q-network with
  a per-job encoder.
- **PPO (alternative):** `models/jobshop_ppo.zip`, MaskablePPO.
- **Rules:** Random, FCFS, SPT, SJF and Priority; every rule chooses
  only eligible jobs.
- **Simple tool models:** `models/simple_dqn.zip`,
  `models/simple_ppo.zip`.
                """
            )


# ==========================================
# PAGE
# ==========================================

web.set_page_config(
    page_title="Production Scheduling Optimizer",
    page_icon="🏭",
    layout="wide"
)

with web.sidebar:

    page = web.radio(
        "Page",
        ("📥 Your data", "🏭 Job-shop demo")
    )

    web.divider()

    web.header("Quick guide")

    web.markdown(
        """
1. Choose **Manual entry** or **Excel upload**.
2. Set the number of identical machines.
3. Select the schedulers to compare.
4. Read the KPIs, Gantt chart and table.

**Priority:** `1` is the highest priority.

**Metrics:** makespan is the finish time of the last job; waiting time
is measured from each job's arrival.
        """
    )

    template_frame = pd.DataFrame(simple_template_rows(10))
    template_buffer = BytesIO()

    with pd.ExcelWriter(template_buffer, engine="openpyxl") as writer:

        template_frame.to_excel(writer, sheet_name="Jobs", index=False)

        instructions = pd.DataFrame({
            "Column": [
                "Job ID", "Arrival Time", "Burst Time", "Priority"
            ],
            "Meaning": [
                "Optional label; jobs are read in row order.",
                "When the job becomes available (>= 0).",
                "Processing time on any machine (> 0).",
                "1 = highest priority (whole number >= 1)."
            ]
        })
        instructions.to_excel(
            writer, sheet_name="Instructions", index=False
        )

    web.download_button(
        "⬇️ Download Excel template",
        template_buffer.getvalue(),
        file_name="scheduling_template.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    web.divider()

    web.caption(
        "Trained models: "
        f"DQN {'✅' if SIMPLE_DQN_PATH.exists() else '❌'} · "
        f"PPO {'✅' if SIMPLE_PPO_PATH.exists() else '❌'}"
    )


web.title("🏭 Production Scheduling Optimizer")
web.caption(
    "Enter your jobs, pick the shop size and compare FCFS, SJF, "
    "priority scheduling, DQN and PPO on exactly the same data."
)

if page == "📥 Your data":
    render_user_page()
else:
    render_jobshop_page()

web.caption(
    "Production Optimization using RL · Shubham (24csu197) · "
    "Rajat Yadav (24csu170) · Dev Kumar (24csu356)"
)
