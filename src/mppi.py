#%%
from ctypes import sizeof
import jax
import numpy as np
import jax.numpy as jnp
from src.rollout import make_vec_rollout_fn, lax_wrapper_step

import matplotlib.pyplot as plt

#%%
def get_candidate_action(scores, all_samples, kappa, standardize=False):

    """
    Each action trajectory gets weighted according to how its reward
    compares to the best score. For the best trajectory, the weight is
    S = np.exp(kappa * 0) = 1. For the remaining ones, the weight should
    tend towards zero quite quickly. A higher value of kappa decreases
    the contribution of the sub-optimal trajectories more quickly.


    scores: (N,)
    all_samples : N, H, act_dim
    kappa: float
    
    Adapted from Nagabandi et al. 2019 Deep Dynamics Models for Learning
    Dexterous Manipulation.
    https://github.com/google-research/pddm/blob/master/pddm/policies/mppi.py

    """
    if standardize:
        scores = (scores - scores.mean()) / scores.std()
    N = scores.shape[0]
    S = jnp.exp(kappa * (scores - jnp.max(scores)))  # (N,)
    denom = jnp.sum(S) + 1e-10

    # Weight all actions of the sequence by that sequence's resulting reward.
    weighted_actions = (all_samples * S.reshape((N, 1, 1))) # (N, H, act_dim)
    mean = np.sum(weighted_actions, axis=0) / denom

    return mean


class MPPI:
    def __init__(self, config):
        self.env_cfg = config["environment"]
        self.ctrl_cfg = config["controller"]
        # phi: terminal cost
        # q: stage cost
        # lambda: temperature
        
        self.lam = self.ctrl_cfg["lambda"] # lambda temperature
        self.act_dim = self.env_cfg["act_dim"]
        self.T = self.ctrl_cfg["T"] # T
        self.K = self.ctrl_cfg["K"] # K
        self.sigma = self.ctrl_cfg["sigma"]
        self.key = jax.random.PRNGKey(self.ctrl_cfg["seed"])
        self.act_max = self.env_cfg["max_torque"]

        self.rollout_fn = self._build_rollout_fn(lax_wrapper_step, self.env_cfg)
        self.reset()

    def reset(self):
        self.nominal_traj = np.zeros((self.T, self.act_dim))

    def _build_rollout_fn(self, step_fn, env_params):
        return make_vec_rollout_fn(step_fn, env_params)


    def _get_action_noise(self):
        self.key, subkey = jax.random.split(self.key)

        noise = jax.random.normal(key=subkey, shape=(
            self.K, self.T, self.act_dim)) * self.sigma

        return noise

    def plot_samples(self):
        samples = self._get_action_noise()
        plt.plot(samples.squeeze().T, alpha=0.1, color="tab:blue")
        plt.show()

    def _ensure_non_zero(self, cost, beta, factor): #UPDATE THIS
        return np.exp(-factor * (cost - beta))

    def get_action(self, obs):

        noise = self._get_action_noise()
        acts = self.nominal_traj + noise
        acts = np.clip(acts, -self.act_max, self.act_max)

        _, rews = self.rollout_fn(obs, acts) # (K, T, 1)

        total_cost = -rews.sum(axis=1).squeeze() # (K,)
        beta = np.min(total_cost) # Min. value over all samples.
        total_cost_non_zero = self._ensure_non_zero(total_cost, beta, 1/self.lam)
        #print(total_cost_non_zero.shape) # (K,)

        eta = np.sum(total_cost_non_zero)
        w = 1/eta * total_cost_non_zero # (K,)


        sol = get_candidate_action(total_cost, acts, kappa=self.lam)
        sol = np.array(sol)
        sol = self.nominal_traj\
            + np.sum(w[:, np.newaxis, np.newaxis] * noise, axis=0)
        self.nominal_traj[:-1] = sol[1:]#np.array(action_traj)
        self.nominal_traj[-1] = sol[-1]

        return sol[0]