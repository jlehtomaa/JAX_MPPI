# JAX_MPPI
Model Predictive Path Integral controller implementation in JAX.

[JAX](https://github.com/google/jax) makes an excellent tool for sampling-based model predictive control (MPC) as it allows to replace for-loops with the efficient [scan](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) procedure, and to vectorize and compile the sampling step for very high efficiency.

This repo contains an implementation of the Model Predictive Path Integral control algorithm (MPPI) by [Williams et al. 2017](https://ieeexplore.ieee.org/document/7989202). Some parts of the code follow the control [library by Shunichi09](https://github.com/Shunichi09/PythonLinearNonlinearControl). The [pendulum](https://gym.openai.com/envs/Pendulum-v0/) environment from the OpenAI gym works as a simple test task to illustrate the the algorithm.

Simply use `python main.py` to run the code. 

