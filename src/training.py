import functools
import pathlib
import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore
import pandas as pd

from src import evaluate, utils
from src.func_approx import FunctionApproximation
from src.game import game
from src.types import TrainingConfig

EVALUATE = True


@dataclass
class StepMetrics:
    n_games: int
    n_frames: int
    n_steps: int
    time_per_step: float
    mean_snake_length: Optional[float] = None
    std_snake_length: Optional[float] = None
    snake_lengths: Optional[jnp.ndarray] = None
    elapsed_time: Optional[float] = None


def record_time(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        start = time.time()
        output = fn(*args, **kwargs)
        return output, time.time() - start

    return wrapped


def initialize_training(
    config: TrainingConfig,
    key: jax.Array,
    init_training_state: Callable,
) -> Tuple[Any, jax.Array]:
    print("Initializing params...")
    key, subkey = jax.random.split(key)
    training_state = init_training_state(subkey)
    return training_state, key


def setup_metrics_file(filename: str) -> None:
    print(f"Creating metrics file: {filename}")
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        f.write("n_game,n_frames,n_steps,time,snake_lenght\n")


def log_training_info(config: TrainingConfig) -> None:
    print(f"Run ID: {config.run_id}")
    print(f"Total training steps: {config.total_training_steps}")
    print(f"Compiled steps: {config.compiled_steps}")
    print(f"Number of snake length evaluations: {config.num_snake_length_evaluations}")
    print(f"Filename: {config.metrics_filename}")
    print("")


def log_step_metrics(metrics: StepMetrics, total_training_steps: int) -> None:
    if metrics.mean_snake_length is None or metrics.std_snake_length is None:
        return

    print(
        f"N steps: {metrics.n_steps:.1e}/{total_training_steps:.1e} - ",
        f"N games: {metrics.n_games:.1e} - ",
        f"N frames: {metrics.n_frames:.1e}\t",
        f"{utils.format_time(metrics.elapsed_time)} ",
        f"({utils.format_time(metrics.time_per_step)} per step) ",
        f"Snake length: {metrics.mean_snake_length:.2f} ± ",
        f"{metrics.std_snake_length:.2f}\t",
    )


def save_metrics_to_file(
    metrics: StepMetrics, filename: str, num_evaluations: int
) -> None:
    if metrics.snake_lengths is None or metrics.elapsed_time is None:
        return

    pd.DataFrame(
        {
            "n_game": [metrics.n_games] * num_evaluations,
            "n_frames": [metrics.n_frames] * num_evaluations,
            "n_steps": [metrics.n_steps] * num_evaluations,
            "time": [metrics.elapsed_time] * num_evaluations,
            "snake_lenght": metrics.snake_lengths,
        }
    ).to_csv(filename, mode="a", header=False, index=False)


def save_checkpoint(run_id: str, training_state: Any, n_frames: str) -> None:
    checkpoint_path = (
        f"agents/{game.GRID_SIZE}x{game.GRID_SIZE}/{run_id}_{n_frames}.pkl"
    )
    pathlib.Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(training_state.params, f)


def evaluate_agent(
    key: jax.Array, params: Any, func_a: FunctionApproximation, num_evaluations: int
) -> Tuple[jnp.ndarray, jax.Array]:
    key, subkey = jax.random.split(key)
    snake_lengths = evaluate.evaluate(subkey, params, func_a, num_evaluations)
    return snake_lengths, key


def collect_step_metrics(
    training_state: Any,
    n_steps: int,
    snake_lengths: jnp.ndarray,
    elapsed_time: float,
) -> StepMetrics:
    time_per_step = elapsed_time / n_steps
    metrics = StepMetrics(
        n_games=training_state.statistics.n_games,
        n_frames=training_state.statistics.n_frames,
        n_steps=n_steps,
        time_per_step=time_per_step,
        elapsed_time=elapsed_time,
    )

    metrics.snake_lengths = snake_lengths
    metrics.mean_snake_length = float(jnp.mean(snake_lengths).squeeze())
    metrics.std_snake_length = float(jnp.std(snake_lengths).squeeze())

    return metrics


def train(
    key: jax.Array,
    loop_fn: Callable,
    init_training_state: Callable,
    approximator: FunctionApproximation,
    opt: optax.GradientTransformation,
    config: TrainingConfig,
):
    log_training_info(config)

    training_state, key = initialize_training(config, key, init_training_state)
    setup_metrics_file(config.metrics_filename)

    print("Starting training...")
    n_steps = 0
    n_steps_last_update = 0

    start_time = time.time()
    while True:
        training_state = loop_fn(training_state, config.compiled_steps)

        n_steps = n_steps + config.compiled_steps

        if EVALUATE:
            (snake_lengths, key), time_to_evaluate = record_time(evaluate_agent)(
                key,
                training_state.params,
                approximator,
                config.num_snake_length_evaluations,
            )
            print(f"time to evaluate: {utils.format_time(time_to_evaluate)}")

            elapsed_time = time.time() - start_time

            metrics = collect_step_metrics(
                training_state,
                n_steps,
                snake_lengths,
                elapsed_time,
            )

            log_step_metrics(metrics, config.total_training_steps)
            save_metrics_to_file(
                metrics,
                config.metrics_filename,
                config.num_snake_length_evaluations,
            )

        if n_steps > n_steps_last_update + config.num_steps_checkpoint:
            save_checkpoint(
                config.run_id, training_state, str(training_state.statistics.n_frames)
            )
            n_steps_last_update = n_steps

        if n_steps > config.total_training_steps:
            break

    save_checkpoint(config.run_id, training_state, "final")

    return training_state
