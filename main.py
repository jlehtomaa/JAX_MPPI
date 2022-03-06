#%%
import gym
import jax.numpy as jnp
from src.utils import read_config
from src.mppi import MPPI
# %%
ID = "pendulum"
CFG_PATH = "./config/" + ID + ".yaml"
config = read_config(CFG_PATH)

mppi = MPPI(config)
env = gym.make("Pendulum-v0")

mppi.reset()
obs, done, rew, n_steps = env.reset(), False, 0.0, 0
obs = obs.reshape((-1,1))

while not done:
    act = mppi.get_action(obs)
    obs, r, done, _ = env.step(act)
    obs = obs.reshape((-1,1))
    rew += r
    n_steps += 1
    #env.render()
