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
    
        
        super().__init__(
            NMPCActor.get_observation_space_from_nmpc(nmpc),    # Dict observation space
            NMPCActor.get_action_space_from_nmpc(nmpc),         # Action space
            lr_schedule, 
            **kwargs,
        )
        self.nmpc = nmpc
        self.register_learnable_params(initial_value_learnable_params)

    @staticmethod
    def get_observation_space_from_nmpc(nmpc: BaseNMPC) -> spaces.Dict:
        obs_space_dict = {"state": spaces.Box(nmpc.x_lb.flatten(), nmpc.x_ub.flatten(), dtype=np.float32)}
        for name, param in nmpc.params.items():
            obs_space_dict[name] = spaces.Box(-np.inf, np.inf, shape=param.shape, dtype=np.float32)
        return spaces.Dict(obs_space_dict) # type: ignore[arg-type]
    
    @staticmethod
    def get_action_space_from_nmpc(nmpc: BaseNMPC) -> spaces.Box:
        return spaces.Box(nmpc.u_lb, nmpc.u_ub)

    def _build(self, lr_schedule: Schedule):
        """
        Builds the optimizer based on learning rate scheduler
        """
        # Handle lr_schedule as either float or callable
        lr = lr_schedule(1) if callable(lr_schedule) else lr_schedule
        
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr, # type: ignore[call-arg]
            **self.optimizer_kwargs
        )

    def forward(self, obs: dict, deterministic: bool = False) -> th.Tensor:
        """
        Forward pass of NMPC with autograd - supports batched observations.
        
        :param obs: Dict observation with 'state' key and non-learnable parameter keys.
                    Can be batched: state shape (batch_size, nx) or unbatched (nx,)
        :param deterministic: Whether to use deterministic actions (ignored for NMPC, always deterministic)
        :return: Actions from NMPC, shape (batch_size, nu) or (nu,)
        """
        # Extract state
        x_0 = obs["state"]
        
        # Determine if batched
        is_batched = x_0.dim() > 1
        batch_size = x_0.shape[0] if is_batched else 1
        
        # Handle batches by solving NMPC for each sample separately
        actions_list = []
        
        for batch_idx in range(batch_size):
            # Extract single sample from batch
            if is_batched:
                x_0_single = x_0[batch_idx]
                nonlearnable_single = {}
                for name in self.nmpc.params.keys():
                    if name in obs:
                        val = obs[name]
                        if isinstance(val, th.Tensor):
                            val = val[batch_idx].detach().cpu().numpy()
                        else:
                            val = val[batch_idx]
                        nonlearnable_single[name] = val
            else:
                x_0_single = x_0
                nonlearnable_single = {}
                for name in self.nmpc.params.keys():
                    if name in obs:
                        val = obs[name]
                        if isinstance(val, th.Tensor):
                            val = val.detach().cpu().numpy()
                        nonlearnable_single[name] = val
            
            # Collect learnable parameters (same for all batch samples)
            learnable_params = {
                name: getattr(self, name) for name in self.nmpc.learnable_params.keys()
            }
            
            # Solve NMPC for this sample
            u_cmd = NMPCAutogradFunction.apply(self.nmpc, x_0_single, learnable_params, nonlearnable_single)
            actions_list.append(u_cmd)
        
        # Stack actions back into batch
        if is_batched:
            return th.stack(actions_list, dim=0)
        else:
            return actions_list[0] # type: ignore
    
    def _predict(self, observation, deterministic: bool = False) -> th.Tensor:
        """Called by SB3 during rollouts."""
        return self.forward(observation, deterministic=deterministic) # type: ignore

    def register_learnable_params(self, initial_value_learnable_params: Dict[str, np.ndarray]) -> None:
        for name in self.nmpc.learnable_params.keys():
            self.register_parameter(name, nn.Parameter(th.Tensor(initial_value_learnable_params[name])))

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return super()._get_constructor_parameters()