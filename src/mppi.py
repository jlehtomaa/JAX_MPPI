import numpy as np
from src.rollout import make_vec_rollout_fn, lax_wrapper_step


class MPPI:
    """ JAX implementation of the MPPI algorithm.

    Williams et al. 2017
    Information Theoretic MPC for Model-Based Reinforcement Learning    
    https://ieeexplore.ieee.org/document/7989202

    Much inspired by the MPPI implementation by Shunichi09, see
    https://github.com/Shunichi09/PythonLinearNonlinearControl

    Assume terminal cost phi(x) = 0.


    """
    def __init__(self, config):
        self.env_cfg = config["environment"]
        self.ctrl_cfg = config["controller"]
        
        self.temperature = self.ctrl_cfg["temperature"] # \lambda
        self.n_samples = self.ctrl_cfg["n_samples"]     # K
        self.n_timesteps = self.ctrl_cfg["n_timesteps"] # T
        self.noise_sigma = self.ctrl_cfg["noise_sigma"] # \Sigma

        self.act_dim = self.env_cfg["act_dim"]
        self.act_max = self.env_cfg["max_torque"]
        self.act_min = -self.env_cfg["max_torque"]
        
        self.rollout_fn = self._build_rollout_fn(lax_wrapper_step, self.env_cfg)
        self.reset()

    def reset(self):
        """Reset the previous control trajectory to zero."""
        self.nominal_traj = np.zeros((self.n_timesteps, self.act_dim))

    def _build_rollout_fn(self, step_fn, env_params):
        """Construct the JAX rollout function.
        
        Arguments:
        ---------
        step_fn: a rollout function that takes as input the current state
                 of the system, and a sequence of noisy control inputs
                 of the shape (n_samples, n_timesteps, act_dim). The rollout
                 function should return a (states, rewards) tuple, 
                 where states has the shape (n_samples, obs_dim), and rewards
                 the shape (n_samples, 1)
        env_params: a dict of parameters consumed by the rollout fn.
        
        """
        return make_vec_rollout_fn(step_fn, env_params)

    def _get_action_noise(self):
        """Get the additive noise applied to the nominal control trajectory."""

        noise = np.random.normal(size=(
            self.n_samples, self.n_timesteps, self.act_dim)) * self.noise_sigma

        return noise

    def _nonzero_cost_fn(self, temp, costs, beta):
        """Temperature-weighted exponential costs, from Williams et al. 2017
        Algorithm 2 MPPI.
        """
        return np.exp(-1. / temp * (costs - beta))

    def get_action(self, obs):
        """ Determine the next optimal action.

        Uses https://github.com/Shunichi09/PythonLinearNonlinearControl ...
        controllers/mppi.py, based on
        Nagabandi et al. (2019). Deep Dynamics Models for Learning
        Dexterous Manipulation. arXiv:1909.11652.

        Arguments:
        ----------
        obs (np.ndarray) : the current state of the system

        Returns:
        --------
        act (np.ndarray): the next optimal control input
        
        """

        noise = self._get_action_noise()
        acts = self.nominal_traj + noise
        acts = np.clip(acts, self.act_min, self.act_max)

        _, rewards = self.rollout_fn(obs, acts) # (K, T, 1)

        # Stage costs from the environment.
        rewards = rewards.sum(axis=1).squeeze()  # (K,)
        exp_rewards = np.exp(self.temperature * (rewards - np.max(rewards)))
        denom = np.sum(exp_rewards) + 1e-10

        weighted_inputs = exp_rewards[:, np.newaxis, np.newaxis] * acts
        sol = np.sum(weighted_inputs, axis=0) / denom

        # Return the first element as an immediate input, and roll the next
        # actions forward by one position.
        self.nominal_traj[:-1] = sol[1:]
        self.nominal_traj[-1] = sol[-1]

        return sol[0]