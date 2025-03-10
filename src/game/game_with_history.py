import jax
import jax.numpy as jnp

from src.game import game


def step(key, state, action):
    next_state, reward = game.step(key, state[0], action)
    return ((next_state, state[0], state[1]), reward)


def random_state(key):
    st = game.random_state(key)
    action = jax.lax.cond(
        st.snake[0][1] <= 2,  # snake head is close to the bottom border
        lambda _: jnp.array([0, 1]),  # move up
        lambda _: jnp.array([0, -1]),  # move down
        0,
    )
    st1, _ = game.step(key, st, action)
    st2, _ = game.step(key, st1, action)
    return (st, st1, st2)
