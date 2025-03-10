import argparse
import pickle
import time

import haiku as hk
import jax
import jax.numpy as jnp

from src import func_approx, models
from src.game import game, game_with_history


def play_game_at_human_speed(key, params):
    key, subkey = jax.random.split(key)
    state = game_with_history.random_state(subkey)

    while True:
        time.sleep(0.05)
        game.draw_game(state[0])
        snake_lenght = game.compute_snake_lenght(state[0].snake)
        # print(f"Snake lenght: {snake_lenght}")
        # print(f"State: {state[0]}")

        state_encoding = approximator.encode_state(state)

        key, subkey = jax.random.split(key)
        action = approximator.deterministic_policy(subkey, state_encoding, params)

        key, subkey = jax.random.split(key)
        state, _ = game_with_history.step(subkey, state, action)

        if state[0].is_over:
            break

    return snake_lenght


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()
    seed = args.seed

    with open(args.agent_file, "rb") as f:
        params = pickle.load(f)

    model = models.get_actor_critic_nn_5x5()
    print("Using Model:")
    print(
        hk.experimental.tabulate(
            model, columns=["module", "input", "output", "params_size", "params_bytes"]
        )(
            func_approx.into_tensor_representation(
                game_with_history.random_state(jax.random.PRNGKey(seed))
            )
        )
    )
    approximator = func_approx.actor_critic_function_aproximation(
        model, func_approx.into_tensor_representation, game.action_to_num
    )

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    play_game_at_human_speed(subkey, params)
