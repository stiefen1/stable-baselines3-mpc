from stable_baselines3.td3.policies import TD3Policy, Actor, ContinuousCritic
from stable_baselines3.common.torch_layers import CombinedExtractor
from typing import Dict
from stable_baselines3.nmpc import BaseNMPC
from stable_baselines3.td3.nmpc_actor import NMPCActor


class TD3NMPCPolicy(TD3Policy):
    """TD3 policy with NMPC actor and neural network critic."""
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        nmpc: BaseNMPC,
        initial_value_learnable_params: Dict,
        **kwargs
    ):
        self.nmpc = nmpc
        self.initial_value_learnable_params = initial_value_learnable_params

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=CombinedExtractor,
            **kwargs
        )
        
    def _build(self, lr_schedule) -> None:
        """Build actor, actor_target, critic, critic_target."""
        
        # Actor: NMPC policy
        self.actor = NMPCActor(
            self.nmpc,
            self.initial_value_learnable_params,
            lr_schedule
        ) # type: ignore
        
        # Actor target: Copy of NMPC policy
        self.actor_target = NMPCActor(
            self.nmpc,
            self.initial_value_learnable_params,
            lr_schedule,
        ) # type: ignore

        # Copy initial weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Create features extractor for critic
        self.features_extractor = self.make_features_extractor()
        features_dim = self.features_extractor.features_dim
        
        # Critic: Standard neural network Q-function (from SB3)
        self.critic = ContinuousCritic(
            self.observation_space,
            self.action_space,                  # type: ignore
            net_arch=self.net_arch["qf"],       
            features_extractor=self.features_extractor,
            features_dim=features_dim,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images
        )
        
        # Critic target: Copy of critic
        self.critic_target = ContinuousCritic(
            self.observation_space,
            self.action_space,                  # type: ignore
            net_arch=self.net_arch["qf"],       
            features_extractor=self.features_extractor,
            features_dim=features_dim,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Setup optimizers
        # Handle lr_schedule as either float or callable
        lr = lr_schedule(1) if callable(lr_schedule) else lr_schedule
        
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr,      # type: ignore
            **self.optimizer_kwargs
        )
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr,      # type: ignore
            **self.optimizer_kwargs
        )