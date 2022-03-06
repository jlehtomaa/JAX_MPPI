#%%
import jax
import jax.numpy as jnp
from src.rollout import make_vec_rollout_fn, lax_wrapper_step

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
    
    """
    if standardize:
        scores = (scores - scores.mean()) / scores.std()
    N = scores.shape[0]
    S = jnp.exp(kappa * (scores - jnp.max(scores)))  # (N,)
    denom = jnp.sum(S) + 1e-10

    # Weight all actions of the sequence by that sequence's resulting reward.
    weighted_actions = (all_samples * S.reshape((N, 1, 1))) # (N, H, act_dim)
    mean = jnp.sum(weighted_actions, axis=0) / denom

    return mean


class MPPI:
    def __init__(self, config):
        self.env_cfg = config["environment"]
        self.ctrl_cfg = config["controller"]
        
        self.kappa = self.ctrl_cfg["kappa"]
        self.act_dim = self.env_cfg["act_dim"]
        self.horizon = self.ctrl_cfg["horizon"]
        self.n_samples = self.ctrl_cfg["n_samples"]
        self.noise_scale = self.ctrl_cfg["noise_scale"]
        self.key = jax.random.PRNGKey(self.ctrl_cfg["seed"])
        self.act_max = self.env_cfg["max_torque"]

        self.rollout_fn = self._build_rollout_fn(lax_wrapper_step, self.env_cfg)
        self.reset()

    def reset(self):
        self.nominal_traj = jnp.zeros((self.horizon, self.act_dim))

    def _build_rollout_fn(self, step_fn, env_params):
        return make_vec_rollout_fn(step_fn, env_params)


    def _get_action_noise(self):
        self.key, subkey = jax.random.split(self.key)
        noise = jax.random.normal(key=subkey, shape=(
            self.n_samples, self.horizon, self.act_dim))
        return noise * self.noise_scale

    def get_action(self, obs):
        
        acts = self.nominal_traj + self._get_action_noise()
        acts = jnp.clip(acts, -self.act_max, self.act_max)
        _, rews = self.rollout_fn(obs, acts)

        total_rews = rews.sum(axis=1).squeeze()
        action_traj = get_candidate_action(total_rews, acts, kappa=self.kappa)
        self.nominal_traj = action_traj
        return action_traj[0]