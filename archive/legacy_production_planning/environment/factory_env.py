
import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
from pathlib import Path


class FactoryEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path=None, episode_length=30):

        super().__init__()

        # LOAD DATASET
        if data_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            data_path = base_dir / "data" / "raw" / "production_data.csv"

        self.df = pd.read_csv(data_path)
        self.episode_length = episode_length

        # PRODUCT CONFIGURATION
        self.products = ["A", "B", "C"]

        self.prices = np.array([500, 400, 300])
        self.costs = np.array([250, 180, 120])

        self.material_required = np.array([3, 2, 1])
        self.hours_required = np.array([2, 1.5, 1])

        self.production_options = [0, 10, 20, 30, 40]

        self.holding_cost = 2
        self.invalid_action_penalty = 2000

        # RESOURCE CONFIGURATION
        self.storage_capacity = 200

        # ACTION SPACE
        self.action_space = spaces.Discrete(125)

        # OBSERVATION SPACE
        # [Demand A, B, C,
        #  Inventory A, B, C,
        #  Raw material, Machine hours, Budget,
        #  Episode progress]

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32
        )

        # INITIAL VARIABLES
        self.inventory = np.zeros(3, dtype=np.float32)

        self.current_day = 0
        self.start_day = 0

        self.raw_material = 0
        self.machine_hours = 0
        self.budget = 0

        self.total_profit = 0.0
        self.daily_demand = np.zeros(3)

    # GET CURRENT DEMAND
    def get_demand(self):

        row = self.df.iloc[self.current_day]

        return np.array([
            row["demand_A"],
            row["demand_B"],
            row["demand_C"]
        ], dtype=np.float32)

    # LOAD DAILY RESOURCES
    def load_resources(self):

        row = self.df.iloc[self.current_day]

        self.raw_material = float(row["raw_material_capacity"])
        self.machine_hours = float(row["machine_hours_capacity"])
        self.budget = float(row["production_budget"])

    # CREATE STATE
    def get_state(self):

        episode_progress = (
            self.current_day - self.start_day
        ) / self.episode_length

        state = np.concatenate([

            self.daily_demand / 60.0,

            self.inventory / self.storage_capacity,

            np.array([
                self.raw_material / 250.0,
                self.machine_hours / 130.0,
                self.budget / 25000.0,
                episode_progress
            ])

        ])

        return np.clip(state, 0.0, 1.0).astype(np.float32)

    # DECODE ACTION
    def decode_action(self, action):

        return np.array(
            np.unravel_index(int(action), (5, 5, 5))
        ) * 10

    # RESET ENVIRONMENT
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        max_start = len(self.df) - self.episode_length

        self.start_day = int(
            self.np_random.integers(0, max_start)
        )

        self.current_day = self.start_day

        self.inventory = np.zeros(3, dtype=np.float32)
        self.total_profit = 0.0

        self.daily_demand = self.get_demand()
        self.load_resources()

        return self.get_state(), {}

    # STEP FUNCTION
    def step(self, action):

        production = self.decode_action(action)

        # RESOURCE CONSUMPTION
        material_used = np.dot(
            production, self.material_required
        )

        hours_used = np.dot(
            production, self.hours_required
        )

        production_cost = np.dot(
            production, self.costs
        )

        available_stock = self.inventory + production

        sales = np.minimum(
            available_stock, self.daily_demand
        )

        ending_inventory = available_stock - sales

        # CHECK FACTORY CONSTRAINTS
        feasible = (
            material_used <= self.raw_material
            and hours_used <= self.machine_hours
            and production_cost <= self.budget
            and ending_inventory.sum() <= self.storage_capacity
        )

        invalid_penalty = 0.0

        # REJECT INFEASIBLE ACTION
        if not feasible:

            production = np.zeros(3)

            material_used = 0.0
            hours_used = 0.0
            production_cost = 0.0

            available_stock = self.inventory.copy()

            sales = np.minimum(
                available_stock, self.daily_demand
            )

            ending_inventory = available_stock - sales

            invalid_penalty = self.invalid_action_penalty

        # REWARD CALCULATION
        revenue = np.dot(sales, self.prices)

        holding_cost = (
            ending_inventory.sum() * self.holding_cost
        )

        unmet_demand = self.daily_demand - sales

        # Profit without an additional shortage penalty
        daily_profit = (
            revenue
            - production_cost
            - holding_cost
            - invalid_penalty
        )

        reward = daily_profit / 1000.0

        self.total_profit += daily_profit

        self.inventory = ending_inventory.astype(np.float32)

        # NEXT DAY
        self.current_day += 1

        terminated = (
            self.current_day >=
            self.start_day + self.episode_length
        )

        truncated = False

        # Only load another day's data if episode continues
        if not terminated:
            self.daily_demand = self.get_demand()
            self.load_resources()

        # INFORMATION
        info = {

            "production": production.copy(),
            "sales": sales.copy(),
            "demand": self.df.iloc[
                self.current_day - 1
            ][
                ["demand_A", "demand_B", "demand_C"]
            ].to_numpy(dtype=float),

            "revenue": float(revenue),
            "production_cost": float(production_cost),
            "holding_cost": float(holding_cost),

            # Retained for compatibility with evaluation scripts
            "shortage_cost": 0.0,
            "unmet_demand": unmet_demand.copy(),

            "daily_profit": float(daily_profit),
            "total_profit": float(self.total_profit),

            "inventory": self.inventory.copy(),

            "material_used": float(material_used),
            "machine_hours_used": float(hours_used),

            "feasible": bool(feasible),
            "invalid_penalty": float(invalid_penalty)
        }

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
            info
        )

    # RENDER ENVIRONMENT
    def render(self):

        print("\n========== FACTORY STATUS ==========")
        print("Day:", self.current_day - self.start_day)
        print("Demand:", self.daily_demand)
        print("Inventory:", self.inventory)
        print("Total Profit:", self.total_profit)
        print("====================================")
