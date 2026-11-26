
import numpy as np
from gymnasium import spaces

from environment.factory_env import FactoryEnv


class FactoryPPOEnv(FactoryEnv):

    def __init__(self, data_path=None, episode_length=30):

        super().__init__(
            data_path=data_path,
            episode_length=episode_length
        )

        # Three independent decisions:
        # Product A: 0, 10, 20, 30, 40
        # Product B: 0, 10, 20, 30, 40
        # Product C: 0, 10, 20, 30, 40

        self.action_space = spaces.MultiDiscrete([5, 5, 5])

    def step(self, action):

        action = np.asarray(action, dtype=np.int64)

        if action.shape != (3,):
            raise ValueError(
                f"Expected action shape (3,), received {action.shape}"
            )

        if np.any(action < 0) or np.any(action >= 5):
            raise ValueError(
                "Each action component must be between 0 and 4."
            )

        # Convert three decisions into the original
        # environment's single discrete action index.
        flat_action = int(
            np.ravel_multi_index(
                tuple(action),
                (5, 5, 5)
            )
        )

        return super().step(flat_action)
