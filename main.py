import gym
from gym.wrappers import Monitor
from src.utils import read_config
from src.mppi import MPPI


CFG_PATH = "./config/pendulum.yaml"
config = read_config(CFG_PATH)

mppi = MPPI(config)
env = gym.make("Pendulum-v0")
#env = Monitor(gym.make("Pendulum-v0"), './video', force=True)

mppi.reset()
obs, done, rew = env.reset(), False, 0.

while not done:
    act = mppi.get_action(obs.reshape((-1,1)))
    obs, r, done, _ = env.step(act)
    rew += r
    env.render()
env.close()
