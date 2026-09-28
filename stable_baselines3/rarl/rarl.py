from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from typing import Any, Dict
import gymnasium as gym

class RARL:
    """ Alternate training of protagonist and adversary models """

    def __init__(
        self,
        protagonist_algorithm: str,
        adversary_algorithm: str,
        protagonist_env: gym.Env,
        adversary_env: gym.Env,
        protagonist_kwargs: Dict[str, Any],
        adversary_kwargs: Dict[str, Any],
        device: str = "cpu",
    ) -> None:

        algorithms = {"PPO": PPO, "SAC": SAC}

        self.protagonist = algorithms[protagonist_algorithm](
            env=protagonist_env,
            **protagonist_kwargs,
        )

        self.adversary = algorithms[adversary_algorithm](
            env=adversary_env,
            **adversary_kwargs,
        )

        # Reach the player environments beneath FlattenObservation/Monitor.
        # Bootstrap the protagonist without adversarial forces. Enable the
        # opponent only once it has completed its first training phase.
        self._protagonist_env = protagonist_env.unwrapped
        self._protagonist_env.adversary = None
        adversary_env.unwrapped.protagonist = self.protagonist

    def learn(
        self,
        protagonist_timesteps: int,
        adversary_timesteps: int,
    ) -> "RARL":
        """Train the protagonist, then the adversary, for one round.

        Each player's timestep argument is its total_timesteps budget per
        phase. The caller controls how many rounds to run. SB3 may exceed a
        budget to finish a rollout (PPO collects n_steps * n_envs transitions).
        Subsequent calls continue with the already trained adversary enabled.
        """
        print("Training protagonist...")
        self.protagonist.set_env(
            self.protagonist.get_env(),
            force_reset=True
        )

        self.protagonist.learn(
            total_timesteps=protagonist_timesteps,
            reset_num_timesteps=False,
            tb_log_name="protagonist",
        )

        print("Training adversary...")
        self.adversary.set_env(
            self.adversary.get_env(),
            force_reset=True,
        )

        self.adversary.learn(
            total_timesteps=adversary_timesteps,
            reset_num_timesteps=False,
            tb_log_name="adversary",
        )
        self._protagonist_env.adversary = self.adversary

        return self
