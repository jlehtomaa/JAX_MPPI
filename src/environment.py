#%%
import numpy as np
import jax
import jax.numpy as jnp
import gym


params = dict(
    max_speed=8.0,
    max_torque=2.0,
    dt=0.05,
    g=9.81,
    m=1.0,
    l=1.0,
)

params["high"] = jnp.array([1.0, 1.0, params["max_speed"]])

#%%

def angle2coords(angle):
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])

def coords2angle(coords):
    cos, sin = coords
    return jnp.arctan2(sin, cos)


def angle_normalize(x):
    """
    https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    """
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi

def step(obs, act, params):
    """ Roll model dynamics forward one step.

    Arguments:
    ---------

    Output:
    -------
    new_state : jnp.ndarray



    ### Action Space
    The action is a `ndarray` with shape `(1,)` representing the torque
    applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ### Observation Space
    The observation is a `ndarray` with shape `(3,)` representing
    the x-y coordinates of the pendulum's free end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ### Starting State
    The starting state is a random angle in *[-pi, pi]* and a random
    angular velocity in *[-1,1]*.
    
    """

    x, y, thdot = obs
    th = coords2angle((x, y))

    g = params["g"]
    l = params["l"]
    m = params["m"]
    dt = params["dt"]
    max_speed = params["max_speed"]

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (act ** 2)

    newthdot = thdot + (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * act) * dt
    newthdot = jnp.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt

    newx, newy = angle2coords(newth)
    newobs = jnp.array([newx, newy, newthdot])

    return newobs, -cost

def lax_step(carry, input, params):
    """ Roll model dynamics forward one step.

    Arguments:
    ---------

    Output:
    -------
    new_carry, outputs
    """
    state = carry
    next_state, reward = step(state, input, params)

    return next_state, (next_state, reward)

from functools import partial
step_fn = partial(lax_step, params=params)
# %%

high = jnp.array([jnp.pi,1, 1])
obs=np.random.uniform(low=-high, high=high)
obs.shape
#%%
obs, rew = step(obs, jnp.array([1.0]), params)
obs

#%%


def make_vec_rollout_fn(step_fn):
    def rollout_fn(obs, act_sequence):
        """
        Arguments:
        ---------
        init_obs : (obs_dim) - shaped array, starting state of sequence.
        act_sequence : (n_samples, act_dim) - shaped array.

        """
        carry = obs
        _, obs_and_rews = jax.lax.scan(f=step_fn, init=carry, xs=act_sequence)

        return obs_and_rews

    func = jax.jit(jax.vmap(rollout_fn, in_axes=(None, 0)))

    return func
# %%
f = make_vec_rollout_fn(step_fn)

#%%
obs.shape
# %%
a, b = f(obs, jnp.ones((4, 10, 1)))
# %%
a.shape
# %%
b.shape
# %%

# %%

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
    denom = np.sum(S) + 1e-10

    # Weight all actions of the sequence by that sequence's resulting reward.
    weighted_actions = (all_samples * S.reshape((N, 1, 1))) # (N, H, act_dim)
    mean = jnp.sum(weighted_actions, axis=0) / denom

    return mean

class MPPI:
    def __init__(self, horizon=30, n_samples=100, seed=0):
        self.key = jax.random.PRNGKey(seed)
        self.act_dim = 1
        self.act_min = -2
        self.act_max = 2
        self.horizon = horizon
        self.n_samples = n_samples
        self.rollout_fn = make_vec_rollout_fn(step_fn)
        self.noise_scale = 0.01
        self.kappa = 10

        self.reset()

    def reset(self):
        self.nominal_traj = jnp.zeros((self.horizon, self.act_dim))


    def _get_action_noise(self):
        self.key, subkey = jax.random.split(self.key)
        noise = jax.random.normal(key=subkey, shape=(
            self.n_samples, self.horizon, self.act_dim))
        return noise * self.noise_scale

    def get_action(self, obs):
        
        acts = self.nominal_traj + self._get_action_noise()
        acts = jnp.clip(acts, self.act_min, self.act_max)
        _, rews = self.rollout_fn(obs, acts)

        total_rews = rews.sum(axis=1).squeeze()
        action_traj = get_candidate_action(total_rews, acts, kappa=self.kappa)
        self.nominal_traj = action_traj
        return action_traj[0]

mppi=MPPI()
#mppi._get_action_noise()
#%%
obs
    

# %%
act =mppi.get_action(obs)
act


#%%
noise = mppi._get_action_noise()

# %%
nominal = jnp.zeros((30, 1))
nominal.shape


#%%
noise.shape

#%%
(nominal+noise).shape



#%%
jnp.ones((4, 1, 1)) * jnp.ones(4)



#%%
env = gym.make("Pendulum-v0")
obs, done, rew, n_steps = env.reset(), False, 0.0, 0
obs=obs.reshape((-1,1))
mppi.reset()
while not done:
    act = mppi.get_action(obs)
    obs, r, done, _ = env.step(act)
    obs=obs.reshape((-1,1))
    rew += r
    n_steps += 1
    env.render()



#%%
rew
# %%
n_steps