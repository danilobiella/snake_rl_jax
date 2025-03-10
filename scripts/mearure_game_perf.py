import time

import jax
import numpy as np

from src.game.game import random_action, random_state, step


def benchmark_game_step(num_steps=1000, warmup_steps=100):
    key = jax.random.PRNGKey(0)
    state = random_state(key)

    print("Warming up JIT compilation...")
    for _ in range(warmup_steps):
        key, subkey = jax.random.split(key)
        action = random_action(subkey)
        state, _ = step(subkey, state, action)

    print(f"Running benchmark with {num_steps} steps...")
    execution_times = []

    for i in range(num_steps):
        key, subkey = jax.random.split(key)
        action = random_action(subkey)

        start_time = time.time()
        state, reward = step(subkey, state, action)
        end_time = time.time()

        execution_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        execution_times.append(execution_time_ms)

        if state.is_over:
            key, subkey = jax.random.split(key)
            state = random_state(subkey)

    avg_time = np.mean(execution_times)
    std_time = np.std(execution_times)

    print(f"Benchmark results ({num_steps} steps):")
    print(f"Execution time: {avg_time:.4f} ± {std_time:.4f} ms")
    print(f"Steps per second: {1000/avg_time:.2f}")

    return execution_times


if __name__ == "__main__":
    benchmark_game_step(num_steps=10000, warmup_steps=100)
