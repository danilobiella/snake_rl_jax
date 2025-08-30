from dataclasses import dataclass

import chex
import jax

Action = jax.Array
Snake = jax.Array
Food = jax.Array


@chex.dataclass
class GameState:
    snake: Snake
    food: Food
    is_over: bool | jax.Array


@chex.dataclass
class TrainingStatistics:
    n_games: int
    n_frames: int
    n_steps: int


@dataclass
class ActorCriticConfig:
    n_agents: int  # Number of agents
    gamma: float  # Discount factor
    beta: float  # Entropy regularization factor
    t_max: float  # number of steps before params update


@dataclass
class TrainingConfig:
    run_id: str  # Unique identifier for the run
    total_training_steps: int  # Number of total training steps to play
    compiled_steps: int  # Number of training steps which are compiled into a JAX for loop
    num_snake_length_evaluations: int  # Number of games to play to evaluate the agent
    metrics_filename: str  # File to save the metrics
    num_steps_checkpoint: int  # Number of steps after which save an agent checkpoint
