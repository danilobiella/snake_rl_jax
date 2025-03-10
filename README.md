# snake_rl_jax


![Alt Text](https://raw.githubusercontent.com/danilobiella/snake_rl_jax/refs/heads/main/gifs/snake_run.gif)

A Jax Implementation of the Advantage Actor Critic (A2C) Reinforcement Learning Algorithm applied to the Snake game. The algorithm is fully implemented in Jax, this has many advantages:

- High performance utilizing Jax's JIT Compilation
- The environment is run on the GPU, further improving performance
- The environments are vectorized and are run in parallel, speeding up training

## How to run

The python environment is handled by [Pixi](https://pixi.sh/latest/)

### Running an already trained agent

    pixi run python play_agent.py --agent agents/5x5/AC2_nn_clip_lr1e-4_55330977.pkl

### Training the agent

    pixi run python train.py --config config.yaml
