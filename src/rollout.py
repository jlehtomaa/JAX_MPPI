#%%
"""

Defines the nominal model used by the controller algorith in terms
of a single step. A single rollout is built by applying the jax.lax.scan
machinery to this step function.

As an example, we use pendulum environment from OpenAI gym:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

The MPPI method, and the scan procedure, easily extend to more complex models,
including those that with time-varying dynamics.

The environment details are:

Action Space:
The action is a `ndarray` with shape `(1,)` representing the torque
applied to free end of the pendulum.

| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

Observation Space:
The observation is a `ndarray` with shape `(3,)` representing
the x-y coordinates of the pendulum's free end and its angular velocity.

| Num | Observation      | Min  | Max |
|-----|------------------|------|-----|
| 0   | x = cos(theta)   | -1.0 | 1.0 |
| 1   | y = sin(angle)   | -1.0 | 1.0 |
| 2   | Angular Velocity | -8.0 | 8.0 |

Starting State:
The starting state is a random angle in *[-pi, pi]* and a random
angular velocity in *[-1,1]*.

"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Iterable, Tuple
# %%
def angle2coords(angle: float) -> jnp.ndarray:
    """ Transforms an angle on a 2D plance to (x, y) coordinates.
    
    """
    return jnp.array([jnp.cos(angle), jnp.sin(angle)])

def coords2angle(coords: Iterable) -> jnp.ndarray:
    """ Transforms (x, y) coordinates into the corresponding vector angle.

    Arguments:
    ----------
    coords : iterable of length 2.
    
    """
    cos, sin = coords
    return jnp.arctan2(sin, cos)


def angle_normalize(x: float) -> float:
    """ Normalize an angle to the [-pi, pi] interval.

    From the Gym repository at :
    /gym/blob/master/gym/envs/classic_control/pendulum.py

    """
    return ((x + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def step(
    obs: jnp.ndarray,
    act: jnp.ndarray,
    params: dict) -> Tuple[jnp.ndarray, float]:
    """ Take one step forward in the pendulum model.

    Follows the Pendulum-v0 model from OpenAI Gym.

    Arguments:
    ---------
    obs : jnp.ndarray of size (3,) The state observation
    act : jnp.ndarray of size (1,) The control input
    params : dict Environment parameters

    Output:
    -------
    newobs : jnp.ndarray of size (3,) New obs after applying act once.
    reward : float Instantaneus reward from the enviornment.
    
    """
    
    x, y, thdot = obs
    th = coords2angle((x, y))
    u = jnp.clip(act, -params["max_torque"], params["max_torque"])

    g = params["g"]
    l = params["l"]
    m = params["m"]
    dt = params["dt"]

    cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    thdot += (3 * g / (2 * l) * jnp.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    thdot = jnp.clip(thdot, -params["max_speed"], params["max_speed"])
    newth = th + thdot * dt

    newx, newy = angle2coords(newth)
    newobs = jnp.array([newx, newy, thdot])

    return newobs, -cost



# def step(
#     obs: jnp.ndarray,
#     act: jnp.ndarray,
#     params: dict) -> Tuple[jnp.ndarray, float]:
    

#     return obs+act, act ** 2



def lax_wrapper_step(carry, input, params):
    """ Wrap a step in the nominal dynamics model to fit the JAX lax scan.

    See https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    for more details.

    Arguments:
    ---------
    carry : tuple The item that lax carries from one step to the next.
            Here, it only contains the current state. More generally, it could
            for instance also have a time counter, or if the nominal model
            also has uncertainty, also a PRNGKey. Intuitively, carry is what
            gets accumulated over the jax.lax.scan call.

    input : array The current control input.

    params : dict Parameters for the nominal environment model.

    Output:
    -------
    new_carry : tuple Updated carry value for the next iteration. Carry must
                have the same shape and dtype across all iterations.

    output : The model outputs we want to track over the scan loop.
    """
    state = carry[0]
    next_state, reward = step(state, input, params)

    new_carry = (next_state, )
    output = (next_state, reward)

    return new_carry, output


def make_vec_rollout_fn(model_step_fn, model_params):

    step_fn = partial(model_step_fn, params=model_params)

    def rollout_fn(obs, act_sequence):
        """
        Arguments:
        ---------
        init_obs : (obs_dim) - shaped array, starting state of sequence.
        act_sequence : (n_samples, act_dim) - shaped array.

        """
        carry = (obs, )
        _, obs_and_rews = jax.lax.scan(f=step_fn, init=carry, xs=act_sequence)

        return obs_and_rews

    func = jax.jit(jax.vmap(rollout_fn, in_axes=(None, 0)))

    return func



# #%%
# # %%
# f = make_vec_rollout_fn(lax_wrapper_step, {})
# # %%
# obs = jnp.zeros(1)
# act = jnp.ones((2, 3, 1)) #K, T, act_dim
# f(obs,act)
# # %%
