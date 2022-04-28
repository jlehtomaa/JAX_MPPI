"""
This module defines the nominal model that the controller has
access to. We write the transition dynamics of the system in terms of
the current state and a single control input. We then formulate the
rollout of N steps by using the jax.lax.scan machinery, which is an
efficient way to get rid of for-loops. Finally, we vectorize the rollout
over K samples of different action trajectories to run the MPPI algorithm.


As a simple test environment, we use Pendulum-v0 from OpenAI gym:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

The MPPI method, and the scan procedure, easily extend to more complex models,
including those that with time-varying dynamics.

The environment details are (from OpenAI gym):

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
    params : dict of environment parameters

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


def lax_wrapper_step(carry, input, params):
    """ Wrapper of a step function.

    This step is not strictly necessary, but makes using the jax.lax.scan
    easier in situations where the dynamics are, say, varying in time,
    or have random components that require the JAX PRNGKey. 

    See https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    for more details.

    Arguments:
    ---------
    carry : tuple The item that lax carries (and accumulates) from one step
            to the next. Here, it only contains the current state.
    input : array The current control input.
    params : dict Parameters for the nominal environment model.

    Output:
    -------
    new_carry : tuple Updated carry value for the next iteration. Carry must
                have the same shape and dtype across all iterations.

    output : The outputs we want to track at each step over the scan loop.
    """
    state = carry[0]
    next_state, reward = step(state, input, params)

    new_carry = (next_state, )
    output = (next_state, reward)

    return new_carry, output


def make_vec_rollout_fn(model_step_fn, model_params):
    """ Build a vectorized call to the rollout function.
    """

    step_fn = partial(model_step_fn, params=model_params)

    def rollout_fn(obs, act_sequence):
        """
        Arguments:
        ---------
        obs : (obs_dim) - shaped array, starting state of sequence.
        act_sequence : (n_steps, act_dim) - shaped array.

        """
        carry = (obs, )
        _, obs_and_rews = jax.lax.scan(f=step_fn, init=carry, xs=act_sequence)

        return obs_and_rews

    # Vectorize the rollout_fn over the first dim of the action sequence.
    # That is, the act_sequence for func should have the following shape:
    # (n_samples, n_steps, act_dim).
    func = jax.jit(jax.vmap(rollout_fn, in_axes=(None, 0)))

    return func