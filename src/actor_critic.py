import dataclasses
from functools import partial
from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
import optax  # type: ignore

from src.func_approx import FunctionApproximation
from src.game import game, game_with_history
from src.types import ActorCriticConfig, GameState, TrainingStatistics


@partial(
    jax.jit,
    static_argnums=(1,),
)
def init_vectorized_state(
    key: jax.Array, n: int
) -> Tuple[GameState, GameState, GameState]:
    keys = jax.random.split(key, n)
    states = [game_with_history.random_state(k) for k in keys]

    return (
        GameState(
            snake=jnp.stack([state[0].snake for state in states]),
            food=jnp.stack([state[0].food for state in states]),
            is_over=jnp.stack(jnp.array([[state[0].is_over for state in states]])).T,
        ),
        GameState(
            snake=jnp.stack([state[1].snake for state in states]),
            food=jnp.stack([state[1].food for state in states]),
            is_over=jnp.stack(jnp.array([[state[1].is_over for state in states]])).T,
        ),
        GameState(
            snake=jnp.stack([state[2].snake for state in states]),
            food=jnp.stack([state[2].food for state in states]),
            is_over=jnp.stack(jnp.array([[state[2].is_over for state in states]])).T,
        ),
    )


@chex.dataclass
class Episode:
    states_encoding: jax.Array
    actions_encoding: jax.Array
    rewards: jax.Array


@chex.dataclass
class TrainingState:
    key: jax.Array
    vectorized_states: Tuple[GameState, GameState, GameState]
    params: jnp.ndarray
    opt_state: optax.OptState
    statistics: TrainingStatistics

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


@partial(
    jax.jit,
    static_argnums=(0,),
)
def null_transition(func_a):
    key = jax.random.PRNGKey(0)
    state = game_with_history.random_state(key)
    state_encoding = func_a.encode_state(state)

    # Random action (not actually random because we don't care about the actual action,
    # only about the tensor shapes
    action = game.random_action(key)
    action_encoding = func_a.encode_action(action)

    # Take action
    next_state, reward = game_with_history.step(key, state, action)

    # Reward is set to -666 to distinguish not assigned states in the memory
    reward = -666

    next_state = (
        GameState(
            snake=next_state[0].snake,
            food=next_state[0].food,
            is_over=jnp.array([True]),
        ),
        GameState(
            snake=next_state[1].snake,
            food=next_state[1].food,
            is_over=jnp.array([True]),
        ),
        GameState(
            snake=next_state[2].snake,
            food=next_state[2].food,
            is_over=jnp.array([True]),
        ),
    )

    return (
        next_state,
        Episode(
            states_encoding=state_encoding,
            actions_encoding=action_encoding,
            rewards=reward,
        ),
        0,
    )


@jax.jit
def random_state_with_right_shape(key):
    state = game_with_history.random_state(key)
    state = (
        GameState(
            snake=state[0].snake,
            food=state[0].food,
            is_over=jnp.array([state[0].is_over]),
        ),
        GameState(
            snake=state[1].snake,
            food=state[1].food,
            is_over=jnp.array([state[1].is_over]),
        ),
        GameState(
            snake=state[2].snake,
            food=state[2].food,
            is_over=jnp.array([state[2].is_over]),
        ),
    )
    return state


@jax.jit
def add_transition(
    memory: Episode,
    transition: Episode,
    n: int,
) -> Episode:
    return jax.tree.map(
        lambda m, t: m.at[n].set(t),
        memory,
        transition,
    )


def get_algo_functions(
    opt: optax.GradientTransformation,
    func_approximation: FunctionApproximation,
    config: ActorCriticConfig,
) -> Tuple[Callable, Callable]:
    print("-" * 30)
    print("Actor-Critic Config:")
    print(f"Number of agents: {config.n_agents}")
    print(f"Gamma: {config.gamma}")
    print(f"Beta: {config.beta}")
    print(f"T max: {config.t_max}")
    print()

    @jax.jit
    def critic_loss_fn(
        params,
        state_encoding,
        target,
    ):
        v = func_approximation.get_v(state_encoding, params)
        error = target - v
        result = jnp.square(error)
        return result

    @jax.jit
    def actor_score_fn(params, state_encoding, action_encoding):
        action_logits = func_approximation.policy_logits(state_encoding, params)
        return jax.nn.log_softmax(action_logits)[action_encoding]

    @jax.jit
    def policy_entropy(params, state_encoding):
        action_logits = func_approximation.policy_logits(state_encoding, params)
        action_probabilities = jax.nn.softmax(action_logits)
        entropy = -1 * jnp.sum(jax.nn.log_softmax(action_logits) * action_probabilities)
        return entropy

    @jax.jit
    def game_step(key, state, params, agent_idx):
        # Handle is_over shape (FIXME)
        state = (
            GameState(
                snake=state[0].snake,
                food=state[0].food,
                is_over=state[0].is_over[0],
            ),
            GameState(
                snake=state[1].snake,
                food=state[1].food,
                is_over=state[1].is_over[0],
            ),
            GameState(
                snake=state[2].snake,
                food=state[2].food,
                is_over=state[2].is_over[0],
            ),
        )

        state_encoding = func_approximation.encode_state(state)

        # Choose action
        key, subkey = jax.random.split(key)
        action = func_approximation.policy(subkey, state_encoding, params)
        action_encoding = func_approximation.encode_action(action)

        # Take action
        key, subkey = jax.random.split(key)
        next_state, reward = game_with_history.step(subkey, state, action)

        # FIXME Handle is_over shape
        next_state = (
            GameState(
                snake=next_state[0].snake,
                food=next_state[0].food,
                is_over=jnp.array([next_state[0].is_over]),
            ),
            GameState(
                snake=next_state[1].snake,
                food=next_state[1].food,
                is_over=jnp.array([next_state[1].is_over]),
            ),
            GameState(
                snake=next_state[2].snake,
                food=next_state[2].food,
                is_over=jnp.array([next_state[2].is_over]),
            ),
        )

        return (
            next_state,
            Episode(
                states_encoding=state_encoding,
                actions_encoding=action_encoding,
                rewards=reward,
            ),
            1,
        )

    @jax.jit
    def scan_fn(carry, memory):
        key, state, params, new_steps_played, agent_idx = carry
        key, subkey = jax.random.split(key)
        next_state, new_transition, delta_steps_played = jax.lax.cond(
            state[0].is_over[0],
            lambda _: null_transition(func_approximation),
            lambda _: game_step(subkey, state, params, agent_idx),
            None,
        )

        return (
            key,
            next_state,
            params,
            new_steps_played + delta_steps_played,
            agent_idx,
        ), new_transition

    @jax.jit
    def apply_gradients(total_vectorized_grads, opt_state, params):
        actor_grads = total_vectorized_grads[0]  # Shape: [n_agents, ...]
        critic_grads = total_vectorized_grads[1]  # Shape: [n_agents, ...]

        actor_grads_avg = jax.tree_map(lambda x: jnp.mean(x, axis=0), actor_grads)
        critic_grads_avg = jax.tree_map(lambda x: jnp.mean(x, axis=0), critic_grads)

        combined_grads = jax.tree_map(
            lambda a, c: a + c, actor_grads_avg, critic_grads_avg
        )

        updates, opt_state = opt.update(combined_grads, opt_state)
        params = optax.apply_updates(params, updates)

        return opt_state, params

    @partial(jax.jit, static_argnums=(3,))
    def play_single_agent(key, state, params, agent_idx):
        # jax.debug.print("playing agent {agent_idx}", agent_idx=agent_idx)
        key, subkey = jax.random.split(key)

        # Play N steps (N=t_max)
        carry = (subkey, state, params, 0, agent_idx)
        final_carry, filled_memory = jax.lax.scan(
            scan_fn, carry, xs=None, length=config.t_max
        )
        key, next_state, params, new_steps_played, agent_idx = final_carry

        # Check if game is over and reset if necessary
        key, subkey = jax.random.split(key)
        next_state, delta_n_games = jax.lax.cond(
            next_state[0].is_over[0],
            lambda _: (random_state_with_right_shape(subkey), 1),
            lambda _: (next_state, 0),
            next_state,
        )

        @jax.jit
        def _compute_grads(episode, total_reward, total_grads, params):
            actor_total_grads, critic_total_grads = total_grads
            total_reward = episode.rewards + config.gamma * total_reward
            # jax.debug.print("reward: {reward}", reward=episode.rewards)
            # jax.debug.print("total_reward 2: {total_reward}", total_reward=total_reward)

            # Compute Critic gradients
            critic_grad_fn = jax.grad(critic_loss_fn)
            critic_grads = critic_grad_fn(
                params,
                episode.states_encoding,
                total_reward,
            )
            # jax.debug.print("critic_grads: {critic_grads}", critic_grads=critic_grads)

            # Compute Actor gradients
            score_function_grads_fn = jax.grad(actor_score_fn)
            entropy_grads_fn = jax.grad(policy_entropy)
            score_function_grads = score_function_grads_fn(
                params, episode.states_encoding, episode.actions_encoding
            )
            v = func_approximation.get_v(episode.states_encoding, params)
            entropy_grads = entropy_grads_fn(params, episode.states_encoding)

            actor_grads = jax.tree.map(
                lambda x, y: -1 * (x * (total_reward - v) + config.beta * y),
                score_function_grads,
                entropy_grads,
            )
            # actor_grads = jax.tree.map(
            #     lambda x: -1 * x * (total_reward - v),
            #     score_function_grads,
            # )

            # Accumulate gradients
            critic_total_grads = jax.tree_map(
                lambda x, y: x + y, critic_total_grads, critic_grads
            )
            actor_total_grads = jax.tree_map(
                lambda x, y: x + y, actor_total_grads, actor_grads
            )

            return total_reward, (actor_total_grads, critic_total_grads)

        @jax.jit
        def compute_grads(carry, episode):
            total_reward, total_grads = carry
            # jax.debug.print("total_reward 1: {total_reward}", total_reward=total_reward)

            # Compute grads if state is valid
            total_reward, total_grads = jax.lax.cond(
                episode.rewards > -600,
                lambda _: _compute_grads(episode, total_reward, total_grads, params),
                lambda _: (total_reward, total_grads),
                None,
            )

            return (total_reward, total_grads), None

        # Initialize total reward to zero if state is terminal, otherwise initialize to
        # Q value of that state
        # jax.debug.print("rewards: {rewards}", rewards=filled_memory.rewards)
        total_reward = jax.lax.cond(
            filled_memory.rewards[-1] > -1,
            lambda _: func_approximation.get_v(
                filled_memory.states_encoding[-1], params
            ),
            lambda _: 0.0,
            None,
        )

        # Compute gradients
        zero_grads = (
            jax.tree_map(lambda x: jnp.zeros_like(x), params),
            jax.tree_map(lambda x: jnp.zeros_like(x), params),
        )
        carry = (total_reward, zero_grads)
        (_, grads), _ = jax.lax.scan(
            compute_grads,
            carry,
            xs=jax.tree.map(lambda x: jnp.flip(x, axis=0), filled_memory),
        )

        return next_state, grads, delta_n_games, new_steps_played

    state_axes = (
        GameState(snake=0, food=0, is_over=0),  # type: ignore
        GameState(snake=0, food=0, is_over=0),  # type: ignore
        GameState(snake=0, food=0, is_over=0),  # type: ignore
    )
    agent_indices = jnp.arange(config.n_agents)
    vectorized_thread_loop = jax.vmap(
        play_single_agent,
        in_axes=(0, state_axes, None, 0),
        out_axes=(state_axes, 0, 0, 0),
    )

    @jax.jit
    def update(training_state):
        key = training_state.key
        vectorized_states = training_state.vectorized_states
        params = training_state.params
        opt_state = training_state.opt_state
        training_stats = training_state.statistics

        # Run vectorized games
        key, subkey = jax.random.split(key)
        vectorized_keys = jax.random.split(subkey, config.n_agents)
        (
            vectorized_states,
            vectorized_grads,
            vec_delta_n_games,
            vec_new_frames_played,
        ) = vectorized_thread_loop(
            vectorized_keys,
            vectorized_states,
            params,
            agent_indices,
        )

        opt_state, params = apply_gradients(vectorized_grads, opt_state, params)

        # Update training stats
        training_stats = TrainingStatistics(
            n_games=training_stats.n_games + vec_delta_n_games.sum(),
            n_frames=training_stats.n_frames
            + config.n_agents * vec_new_frames_played.sum(),
            n_steps=training_stats.n_steps + 1,
        )

        return TrainingState(
            key=key,
            vectorized_states=vectorized_states,
            params=params,
            opt_state=opt_state,
            statistics=training_stats,
        )

    @partial(jax.jit, static_argnums=(1,))
    def loop(training_state, n_iter: int):
        return jax.lax.fori_loop(0, n_iter, lambda i, ts: update(ts), training_state)

    @jax.jit
    def init_training_state(key):
        key, subkey = jax.random.split(key)
        params = func_approximation.init_params(subkey)

        key, subkey = jax.random.split(key)
        vectorized_states = init_vectorized_state(subkey, config.n_agents)

        # Initialize training state
        key, subkey = jax.random.split(key)

        # Build training state
        training_state = TrainingState(
            key=subkey,
            vectorized_states=vectorized_states,
            params=params,
            opt_state=opt.init(params),
            statistics=TrainingStatistics(n_games=0, n_frames=1, n_steps=1),
        )
        return training_state

    return loop, init_training_state
