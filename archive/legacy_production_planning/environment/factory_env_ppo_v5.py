
from environment.factory_env_ppo_v4 import FactoryPPOV4Env


class FactoryPPOV5Env(FactoryPPOV4Env):
    """
    PPO V5 environment.

    Retains V4's factory constraints and action masking.
    No changes to reward calculation or production economics.
    """

    def __init__(self, data_path=None, episode_length=30):
        super().__init__(
            data_path=data_path,
            episode_length=episode_length
        )
