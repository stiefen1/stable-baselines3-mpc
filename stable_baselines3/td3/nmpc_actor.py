from typing import Any, Union, Dict

import torch as th, numpy.typing as npt, numpy as np
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.nmpc import BaseNMPC
from stable_baselines3.nmpc import NMPCAutogradFunction


class NMPCActor(BasePolicy):
    """
    Nonlinear Model Predicive Control Actor (policy).

    :param nmpc: A Nonlinear Model Predictive Control object with learnable and non-learnable parameters
    :param initial_value_learnable_params: Initial values for learnable parameters
    :param lr_schedule: Learning rate scheduler or constant
    """

    def __init__(
        self,
        nmpc: BaseNMPC,
        initial_value_learnable_params: Dict[str, npt.NDArray],
        lr_schedule: Union[float, Schedule] = 1e-3,
        **kwargs
    ):
        assert len(nmpc.learnable_params) > 0, f"NMPC object must have learnable parameters. Got nmpc.learnable_params = {nmpc.learnable_params}"
        
        # Create observation space as Dict: state + non-learnable parameters
        obs_space_dict = {"state": spaces.Box(nmpc.x_lb.flatten(), nmpc.x_ub.flatten(), dtype=np.float32)}
        for name, shape in nmpc.params.items():
            obs_space_dict[name] = spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
        obs_space = spaces.Dict(obs_space_dict) # type: ignore[arg-type]
        
        super().__init__(
            obs_space,                          # Dict observation space
            spaces.Box(nmpc.u_lb, nmpc.u_ub),   # Action space
            lr_schedule, 
            **kwargs,
        )
        self.nmpc = nmpc
        self.register_learnable_params(initial_value_learnable_params)

    def _build(self, lr_schedule: Schedule):
        """
        Builds the optimizer based on learning rate scheduler
        """
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1), # type: ignore[call-arg]
            **self.optimizer_kwargs
        )

    def forward(self, obs: dict) -> th.Tensor:
        """
        Forward pass of NMPC with autograd.
        
        :param obs: Dict observation with 'state' key and non-learnable parameter keys
        :return: Action from NMPC
        """
        # Extract state
        x_0 = obs["state"]
        
        # Extract non-learnable parameters from observation
        nonlearnable_params = {}
        for name in self.nmpc.params.keys():
            if name in obs:
                val = obs[name]
                # Convert torch tensor to numpy if needed
                if isinstance(val, th.Tensor):
                    val = val.detach().cpu().numpy()
                nonlearnable_params[name] = val
        
        # Collect learnable parameters
        learnable_params = {
            name: getattr(self, name) for name in self.nmpc.learnable_params.keys()
        }

        # Apply custom autograd function
        x_0_tensor = x_0.squeeze(0) if x_0.dim() > 1 else x_0
        u_cmd = NMPCAutogradFunction.apply(self.nmpc, x_0_tensor, learnable_params, nonlearnable_params)
        return u_cmd # type: ignore
    
    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        """Called by SB3 during rollouts."""
        return self.forward(observation) # type: ignore

    def register_learnable_params(self, initial_value_learnable_params: Dict[str, np.ndarray]) -> None:
        for name in self.nmpc.learnable_params.keys():
            self.register_parameter(name, nn.Parameter(th.Tensor(initial_value_learnable_params[name])))

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return super()._get_constructor_parameters()