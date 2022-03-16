#%%
import numpy as np
import gym

from rollout import step

# %%
env = gym.make("Pendulum-v0")
# %%
obs_ = env.reset()
act = np.array([1.235])
#%%
for i in range(5):
    obs, rew, done, info = env.step(act)
    print(rew)
# %%
params = dict(
  seed= 0,
  max_speed= 8.0,
  max_torque= 2.0,
  dt= 0.05,
  g= 10.0,
  m= 1.0,
  l= 1.0,
  obs_max= [1.0, 1.0, 8.0],
  act_dim= 1,
  obs_dim= 3,
)
# %%
for i in range(5):
    obs_, rew_ = step(obs_, act, params)
    print(rew_.T)

# %%
