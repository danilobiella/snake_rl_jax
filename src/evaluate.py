from functools import partial

import jax
import jax.numpy as jnp

from src.game import game, game_with_history


@partial(jax.jit, static_argnums=(2,))
def play_game(key, params, approximator):
    @jax.jit
    def step(args):
        state, snake_lenght, key, i = args

        state_encoding = approximator.encode_state(state)

        key, subkey = jax.random.split(key)
        action = approximator.deterministic_policy(subkey, state_encoding, params)

        key, subkey = jax.random.split(key)
        next_state, reward = game_with_history.step(subkey, state, action)

        snake_lenght = game.compute_snake_lenght(state[0].snake)

        return next_state, snake_lenght, key, i + 1

    key, subkey = jax.random.split(key)
    state = game_with_history.random_state(subkey)
    state_encoding = approximator.encode_state(state)

    key, subkey = jax.random.split(key)
    action = approximator.deterministic_policy(subkey, state_encoding, params)

    key, subkey = jax.random.split(key)
    next_state, reward = game_with_history.step(subkey, state, action)

    key, subkey = jax.random.split(key)
    _, snake_lenght, _, _ = jax.lax.while_loop(
        lambda args: jnp.logical_and(
            jnp.logical_not(args[0][0].is_over), args[-1] < 10_000
        ),
        step,
        (next_state, 0, subkey, 0),
    )

    return snake_lenght


@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
    ),
)
def evaluate(key, weights, approximator, n_runs=100):
    keys = jax.random.split(key, n_runs)
    out = jax.vmap(play_game, in_axes=(0, None, None))(keys, weights, approximator)
    return out
