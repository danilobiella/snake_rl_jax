import functools
from typing import Callable, Protocol

import haiku as hk
import jax
import jax.numpy as jnp

from src.game import game, game_with_history
from src.types import Action, GameState


class FunctionApproximation(Protocol):
    def get_v(self, state, params): ...

    def policy(self, key, state, params): ...

    def init_params(self, key): ...

    def encode_state(self, state): ...

    def encode_action(self, action): ...


def actor_critic_function_aproximation(
    model: hk.Transformed,
    encode_state: Callable[[GameState], jax.Array],
    encode_action: Callable[[Action], jax.Array],
) -> FunctionApproximation:
    class ActorCriticApproximation(FunctionApproximation):
        @functools.partial(jax.jit, static_argnums=(0,))
        def get_v(self, state_encoding, params):
            res = model.apply(params, None, state_encoding)
            return res[1][0]

        @functools.partial(jax.jit, static_argnums=(0,))
        def policy(self, key, state_encoding, params):
            action_logits = self.policy_logits(state_encoding, params)
            return game.num_to_action(jax.random.categorical(key, action_logits))

        @functools.partial(jax.jit, static_argnums=(0,))
        def deterministic_policy(self, key, state_encoding, params):
            action_logits = self.policy_logits(state_encoding, params)
            return game.num_to_action(jnp.argmax(action_logits))

        @functools.partial(jax.jit, static_argnums=(0,))
        def policy_logits(self, state_encoding, params):
            return model.apply(params, None, state_encoding)[0]

        @functools.partial(jax.jit, static_argnums=(0,))
        def init_params(self, key):
            key1, key2 = jax.random.split(key)
            return model.init(key1, encode_state(game_with_history.random_state(key2)))

        @functools.partial(jax.jit, static_argnums=(0,))
        def encode_state(self, state):
            return encode_state(state)

        @functools.partial(jax.jit, static_argnums=(0,))
        def encode_action(self, action):
            return encode_action(action)

    return ActorCriticApproximation()


@jax.jit
def init_weights():
    return jnp.zeros((14, 4))


@jax.jit
def features(state_history):
    state, _, _ = state_history
    snake, food = state.snake, state.food
    # jax.debug.print("snake: {snake}", snake=snake)
    # jax.debug.print("food: {food}", food=food)
    vector = jnp.array(
        [
            # y differences
            jnp.abs(snake[0][0] - food[0]),
            # x differences
            jnp.abs(snake[0][1] - food[1]),
            # is food up
            snake[0][0] > food[0],
            # is food down
            snake[0][0] < food[0],
            # is food left
            snake[0][1] > food[1],
            # is food right
            snake[0][1] < food[1],
            # is left border
            snake[0][0] == 0,
            # is right border
            snake[0][0] == game.GRID_SIZE - 1,
            # is top border
            snake[0][1] == 0,
            # is bottom border
            snake[0][1] == game.GRID_SIZE - 1,
            # is top body
            jnp.all(snake[0] + jnp.array([0, 1]) == snake[1:], axis=1).any(),
            # is right body
            jnp.all(snake[0] + jnp.array([1, 0]) == snake[1:], axis=1).any(),
            # is bottom body
            jnp.all(snake[0] + jnp.array([0, -1]) == snake[1:], axis=1).any(),
            # is left body
            jnp.all(snake[0] + jnp.array([-1, 0]) == snake[1:], axis=1).any(),
        ],
        dtype=jnp.float32,
    )
    # jax.debug.print("vector: {vector}", vector=vector)
    return vector


@jax.jit
def into_tensor_representation(state_history):
    def extract_body_rep(state):
        grid_size = game.GRID_SIZE
        rep = jnp.zeros((grid_size, grid_size))

        def body_fun(i, rep):
            s = state.snake[i]
            is_valid = jnp.all(s >= 0)
            return rep.at[tuple(s)].set(jnp.where(is_valid, 1, rep[tuple(s)]))

        rep = jax.lax.fori_loop(0, len(state.snake), body_fun, rep)
        return rep

    def extract_food_rep(state):
        return jnp.zeros((game.GRID_SIZE, game.GRID_SIZE)).at[tuple(state.food)].set(1)

    def extract_both_rep(state):
        rep = extract_body_rep(state)
        return rep.at[tuple(state.food)].set(-1)

    return jnp.stack(
        [
            extract_both_rep(state_history[0]),
            extract_both_rep(state_history[1]),
            extract_both_rep(state_history[2]),
        ]
    ).transpose((2, 1, 0))
