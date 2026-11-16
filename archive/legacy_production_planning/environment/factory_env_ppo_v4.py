
import numpy as np
from itertools import product

from environment.factory_env import FactoryEnv


class FactoryPPOV4Env(FactoryEnv):

    def __init__(self, data_path=None, episode_length=30):
        super().__init__(
            data_path=data_path,
            episode_length=episode_length
        )

        # All 125 possible production combinations.
        self.production_combinations = np.array(
            list(product([0, 10, 20, 30, 40], repeat=3)),
            dtype=np.float32
        )

    def action_masks(self):
        """
        Return a Boolean mask:
        True  = feasible action
        False = infeasible action
        """

        masks = []

        for production in self.production_combinations:

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

            sales = np.minimum(
                available_stock,
                self.daily_demand
            )

            ending_inventory = available_stock - sales

            feasible = (
                material_used <= self.raw_material
                and hours_used <= self.machine_hours
                and production_cost <= self.budget
                and ending_inventory.sum() <= self.storage_capacity
            )

            masks.append(feasible)

        masks = np.array(masks, dtype=bool)

        # There should always be at least one feasible action.
        if not masks.any():
            raise RuntimeError("No feasible production action found.")

        return masks
