
from environment.factory_env_ppo_v5 import FactoryPPOV5Env


class FactoryPPOV6Env(FactoryPPOV5Env):
    """
    PPO V6 environment.

    Retains V4/V5's factory constraints and action masking.
    No changes to reward calculation or production economics.

    V6 is a hyperparameter experiment only: the V5 diagnostic
    showed an under-trained policy (near-zero approx_kl and
    explained variance), so V6 restores a standard learning
    rate, lowers the entropy coefficient and trains longer.
    """

    def __init__(self, data_path=None, episode_length=30):
        super().__init__(
            data_path=data_path,
            episode_length=episode_length
        )
