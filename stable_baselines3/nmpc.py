from typing import Literal, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import warnings

import casadi as cs, numpy.typing as npt, numpy as np
from csnlp.wrappers import Mpc
from csnlp import Nlp, wrappers

DEFAULT_IPOPT_SOLVER_OPTS = {
            "error_on_fail": False,  # Don't crash on solver failure
            "expand": True,
            "print_time": False,
            "record_time": True,
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,  # Increased iterations
            "ipopt.tol": 1e-6,      # Relaxed tolerance
            "ipopt.acceptable_tol": 1e-4,  # Acceptable solution tolerance
            "ipopt.mu_init": 1e-3,  # Initial barrier parameter
            "ipopt.warm_start_init_point": "no",  # Use warm start
        }

class BaseNMPC(Mpc, ABC):
    """
    The base nmpc object: generate control commands based on observations. 

    :param dynamics: Discrete dynamics of the system to control
    :param horizon: The prediction horizon of the NMPC
    :param nx: State dimension
    :param nu: Control command dimension
    :param learnable_params: Dictionnary mapping names of learnable parameters to their shape
    :param params: Dictionnary mapping names of non-learnable parameters to their shape
    :param shooting: Shooting method to discretize the OCP
    :param solver: Name of the solver to use
    :param solver_opts: Options for the selected solver
    """

    def __init__(
            self,
            dynamics: cs.Function,
            horizon: int,
            nx: int,
            nu: int, 
            u_lb: npt.NDArray,
            u_ub: npt.NDArray,
            x_lb: Optional[npt.NDArray] = None,
            x_ub: Optional[npt.NDArray] = None,
            learnable_params: Optional[Dict[str, Tuple]] = None,
            params: Optional[Dict[str, Tuple]] = None,
            shooting: Literal["single", "multi"] = "multi",
            solver: Literal["ipopt"] = "ipopt",
            solver_opts: Dict = DEFAULT_IPOPT_SOLVER_OPTS

    ):
        Mpc.__init__(self, Nlp[cs.SX](), prediction_horizon=horizon, shooting=shooting)
        self.dynamics = dynamics
        self.horizon = horizon
        self.nx = nx
        self.nu = nu
        self.u_lb = u_lb.reshape(-1, 1)
        self.u_ub = u_ub.reshape(-1, 1)
        self.x_lb = x_lb.reshape(-1, 1) if x_lb is not None else np.array(self.nx * [-np.inf])
        self.x_ub = x_ub.reshape(-1, 1) if x_ub is not None else np.array(self.nx * [np.inf])
        self.prev_sol = None

        # Store learnable parameters in a dict. Needed to discriminate learnable from non-learnable ones
        self.learnable_params = {}
        learnable_params = learnable_params if learnable_params is not None else {}
        for name, shape in learnable_params.items():
            self.learnable_params.update({name: self.nlp.parameter(name, shape)})
            print(f"Added {name} as learnable parameter with shape {shape} to {type(self).__name__} object.")

        self.params = {}
        params = params if params is not None else {}
        for name, shape in params.items():
            self.params.update({name: self.nlp.parameter(name, shape)})
            print(f"Added {name} as non-learnable parameter with shape {shape} to {type(self).__name__} object.")

        self.shooting = shooting

        assert self.u_lb.shape[0] == self.nu, f"Lower bound of u must have its first dimension equal to nu={self.nu}, but self.u_lb.shape = {self.u_lb.shape}."
        assert self.u_ub.shape[0] == self.nu, f"Upper bound of u must have its first dimension equal to nu={self.nu}, but self.u_ub.shape = {self.u_ub.shape}."
        assert self.x_lb.shape[0] == self.nx, f"Lower bound of x must have its first dimension equal to nx={self.nx}, but self.x_lb.shape = {self.x_lb.shape}."
        assert self.x_ub.shape[0] == self.nx, f"Upper bound of x must have its first dimension equal to nx={self.nx}, but self.x_ub.shape = {self.x_ub.shape}."

        self.init_nlp(solver, solver_opts)

    def init_nlp(self, solver: Literal["ipopt"], solver_opts: Dict) -> None:
        # Create state symbol
        x, _ = self.state(
            "x",
            self.nx,
            lb=self.x_lb.reshape(-1, 1),
            ub=self.x_ub.reshape(-1, 1)
        )
        assert x is not None, f"x was not defined correctly. x={x}"

        # Create control command symbol
        u, _ = self.action(
            "u",
            self.nu,
            lb=self.u_lb.reshape(-1, 1),
            ub=self.u_ub.reshape(-1, 1)
        )
        assert u is not None, f"u was not defined correctly. u={u}"

        self.set_nonlinear_dynamics(self.dynamics)
        self.minimize(self.cost_function(x, u))
        self.init_solver(solver_opts, solver)


    @abstractmethod
    def lagrange(self, xk: cs.SX, uk: cs.SX, k: int, *args, **kwargs) -> cs.SX:
        """
        learnable/non-learnable parameters can be used here by accessing its name:

        self.learnable_params["Q"]
        """
        pass

    @abstractmethod
    def mayer(self, xN: cs.SX, *args, **kwargs) -> cs.SX:
        """
        learnable/non-learnable parameters can be used here by accessing its name:

        self.params["wpts"]
        """
        pass

    def cost_function(self, x: cs.SX, u: cs.SX, *args, **kwargs) -> cs.SX:
        """
        :param: args: List of arguments whose value can change along horizon
        :params: kwargs: Dict of argument constant over the horizon
        """
        cost: cs.SX = cs.SX(0)
        for k in range(self.horizon):
            cost += self.lagrange(x[:, k], u[:, k], k, *args, **kwargs)
        
        cost += self.mayer(x[:, self.horizon], self.horizon, *args, **kwargs)
        return cost

    def forward(self, x_0: npt.NDArray, nonlearnable_params: Dict[str, npt.NDArray], learnable_params: Dict[str, npt.NDArray], *args, **kwargs) -> npt.NDArray:
        """
        Forward pass of NMPC, i.e. solve and returns first control commands. Only the non-learnable parameters
        must be provided as input. 
        """
        
        sol = self.solve(
            pars={"x_0": x_0.reshape(self.nx, 1)} | nonlearnable_params | learnable_params,
            *args,
            **kwargs
        )

        # Extract first control commands
        u_cmd = sol.vals["u"][:, 0].full().flatten()

        # Save solution as previous one
        self.prev_sol = sol

        return u_cmd
    
    def get_mpc_sensitivities(self) -> npt.NDArray:
        """
        Returns sensitivites of the action w.r.t learnable params.
        """
        assert self.prev_sol is not None, f"Previous solution is none."
        if len(self.learnable_params) > 0:
            mpc_ = wrappers.NlpSensitivity(self, cs.horzcat(*list(self.learnable_params.values()))) # type: ignore
            return mpc_.parametric_sensitivity(self._actions["u"][:, 0], self.prev_sol, second_order=False)[0].full()
        else:
            warnings.warn(f"No learnable parameters provided: Unable to compute MPC sensitivites.")
            return np.array([])

    def learnable_params_to_vector(self, learnable_dict: Dict[str, npt.NDArray]) -> npt.NDArray:
        """Convert learnable params dict to flat vector (order preserved)."""
        return np.concatenate([v.flatten() for v in learnable_dict.values()])

    def vector_to_learnable_params(self, vector: npt.NDArray) -> Dict[str, npt.NDArray]:
        """Convert flat vector back to learnable params dict."""
        # Reconstruct using stored shapes and insertion order
        offset = 0
        result = {}
        for name in self.learnable_params.keys():  # Order preserved
            shape = self.learnable_params[name].shape
            size = np.prod(shape) # shape[0] * shape[1] * ... * shape[N-1]
            result[name] = vector[offset:offset+size].reshape(shape)
            offset += size
        return result

