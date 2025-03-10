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
    num_screens_to_play: int  # Number of screens to play
    loops_before_metrics: int  # Number of loops before displaying metrics
    num_snake_length_evaluations: int  # Number of games to play to evaluate the agent
    metrics_filename: str  # File to save the metrics
    num_games_checkpoint: int  # Number of games after which save an agent checkpoint
