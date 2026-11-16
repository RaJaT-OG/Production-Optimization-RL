
import numpy as np
from itertools import product
from gymnasium import spaces

from environment.factory_env import FactoryEnv


class FactoryPPOV3Env(FactoryEnv):

    def __init__(self, data_path=None, episode_length=30):
        super().__init__(
            data_path=data_path,
            episode_length=episode_length
        )

        # Three independent production decisions:
        # Each product: 0, 10, 20, 30, or 40 units
        self.action_space = spaces.Discrete(125)

        self.production_combinations = np.array(
            list(product([0, 10, 20, 30, 40], repeat=3)),
            dtype=np.float32
        )

    def _calculate_candidate(self, production):
        """Calculate feasibility and profit for a production combination."""

        material_used = float(
            np.dot(production, self.material_required)
        )
        hours_used = float(
            np.dot(production, self.hours_required)
        )
        production_cost = float(
            np.dot(production, self.costs)
        )

        available_stock = self.inventory + production
        sales = np.minimum(available_stock, self.daily_demand)
        ending_inventory = available_stock - sales

        feasible = (
            material_used <= self.raw_material
            and hours_used <= self.machine_hours
            and production_cost <= self.budget
            and ending_inventory.sum() <= self.storage_capacity
        )

        revenue = float(np.dot(sales, self.prices))
        holding_cost = float(
            ending_inventory.sum() * self.holding_cost
        )

        daily_profit = revenue - production_cost - holding_cost

        return {
            "feasible": bool(feasible),
            "material_used": material_used,
            "hours_used": hours_used,
            "production_cost": production_cost,
            "sales": sales,
            "ending_inventory": ending_inventory,
            "revenue": revenue,
            "holding_cost": holding_cost,
            "daily_profit": daily_profit
        }

    def step(self, action):

        # PPO V3 uses the original 0-124 discrete action indexing.
        requested_production = self.decode_action(action)

        requested_result = self._calculate_candidate(
            requested_production
        )

        # Keep the requested action if it is feasible.
        if requested_result["feasible"]:
            production = requested_production
            result = requested_result

        else:
            # Otherwise, find the closest feasible production combination.
            # If multiple combinations are equally close, choose the one
            # with the highest immediate profit.
            best_candidate = None
            best_key = None

            for candidate in self.production_combinations:

                candidate_result = self._calculate_candidate(candidate)

                if not candidate_result["feasible"]:
                    continue

                distance = float(
                    np.sum((candidate - requested_production) ** 2)
                )

                key = (
                    distance,
                    -candidate_result["daily_profit"]
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best_candidate = candidate.copy()
                    best_candidate_result = candidate_result

            if best_candidate is None:
                raise RuntimeError(
                    "No feasible production combination was found."
                )

            production = best_candidate
            result = best_candidate_result

        # Apply the selected production decision.
        sales = result["sales"]
        ending_inventory = result["ending_inventory"]

        revenue = result["revenue"]
        production_cost = result["production_cost"]
        holding_cost = result["holding_cost"]
        daily_profit = result["daily_profit"]

        material_used = result["material_used"]
        hours_used = result["hours_used"]

        unmet_demand = self.daily_demand - sales
        demand_today = self.daily_demand.copy()

        # Reward remains directly connected to actual profit.
        reward = daily_profit / 1000.0

        self.total_profit += daily_profit
        self.inventory = ending_inventory.astype(np.float32)

        # Advance to the next day.
        self.current_day += 1

        terminated = (
            self.current_day >=
            self.start_day + self.episode_length
        )
        truncated = False

        if not terminated:
            self.daily_demand = self.get_demand()
            self.load_resources()

        info = {
            "production": production.copy(),
            "requested_production": requested_production.copy(),
            "sales": sales.copy(),
            "demand": demand_today,

            "revenue": float(revenue),
            "production_cost": float(production_cost),
            "holding_cost": float(holding_cost),

            "shortage_cost": 0.0,
            "unmet_demand": unmet_demand.copy(),

            "daily_profit": float(daily_profit),
            "total_profit": float(self.total_profit),

            "inventory": self.inventory.copy(),

            "material_used": float(material_used),
            "machine_hours_used": float(hours_used),

            # False means the requested action needed adjustment.
            "feasible": bool(requested_result["feasible"]),
            "action_adjusted": bool(
                not requested_result["feasible"]
            ),
            "invalid_penalty": 0.0
        }

        return (
            self.get_state(),
            float(reward),
            terminated,
            truncated,
            info
        )
