import jax
import jax.numpy as jnp

from src.types import Action, Food, GameState, Snake

GRID_SIZE = 5

ACTIONS = jnp.array(
    [
        [0, 1],  # 0: RIGHT
        [0, -1],  # 1: LEFT
        [1, 0],  # 2: DOWN
        [-1, 0],  # 3: UP
    ]
)


@jax.jit
def step(key: jax.Array, state: GameState, action: Action) -> tuple[GameState, float]:
    snake, food = state.snake, state.food
    moved_snake = _move_snake(snake, action)

    key, subkey = jax.random.split(key)
    state, reward = jax.lax.cond(
        _snake_crashed(moved_snake),
        lambda state: (GameState(snake=moved_snake, food=state.food, is_over=True), -1),
        lambda state: jax.lax.cond(
            _ate_food(moved_snake, food),
            lambda state: (
                GameState(
                    snake=_grow_snake(snake, moved_snake),
                    food=_generate_food(subkey, _grow_snake(snake, moved_snake)),
                    is_over=False,
                ),
                1,
            ),
            lambda state: (
                GameState(snake=moved_snake, food=state.food, is_over=False),
                0,
            ),
            state,
        ),
        state,
    )

    return state, reward


@jax.jit
def random_state(key: jax.Array) -> GameState:
    key1, key2 = jax.random.split(key)
    snake = jnp.ones((GRID_SIZE * GRID_SIZE, 2), jnp.int32) * -1
    snake_head = jax.random.randint(key1, (2,), 0, GRID_SIZE)
    snake = snake.at[0].set(snake_head)
    food = _generate_food(key2, snake)
    return GameState(snake=snake, food=food, is_over=False)


@jax.jit
def empty_board() -> GameState:
    snake = jnp.ones((GRID_SIZE * GRID_SIZE, 2), jnp.int32) * -1
    food = jnp.array([-1, -1])
    return GameState(snake=snake, food=food, is_over=False)


@jax.jit
def random_action(key):
    n = jax.random.randint(key, (1,), 0, 4)
    return num_to_action(n[0])


@jax.jit
def compute_snake_lenght(snake: Snake) -> jax.Array:
    return jnp.sum(jnp.any(snake != -1, axis=1))


@jax.jit
def action_to_num(action: jax.Array) -> int:
    matches = jnp.all(action == ACTIONS, axis=1)
    return jnp.where(matches.any(), jnp.argmax(matches), -1)


@jax.jit
def num_to_action(num: int) -> Action:
    return ACTIONS[num]


def draw_game(state: GameState, clear_screen=True):
    snake, food = state.snake, state.food
    grid = [["⬛" for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    grid[food[0]][food[1]] = "🍎"

    for snake_piece in snake:
        if jnp.all(snake_piece < 0):
            break

        grid[snake_piece[0]][snake_piece[1]] = "🟩"

    grid_str = "\n".join(["".join(row) for row in grid])

    if clear_screen:
        print("\033[H\033[J")

    print(f"{grid_str}")


def _generate_food(key: jax.Array, snake: Snake) -> Food:
    # FIXME: Must be a better way than doing this random thing
    key, new_food = _generate_food_step(key)
    snake_not_at_max_lenght = compute_snake_lenght(snake) < GRID_SIZE * GRID_SIZE

    _, food = jax.lax.while_loop(
        lambda args: jnp.any(jnp.all(snake == args[1], axis=1))
        & snake_not_at_max_lenght,
        lambda args: _generate_food_step(args[0]),
        (key, new_food),
    )
    return food


@jax.jit
def _generate_food_step(key: jax.Array) -> tuple[jax.Array, Food]:
    key, subkey = jax.random.split(key)
    new_food = jax.random.randint(subkey, (2,), 0, GRID_SIZE)
    return key, new_food


@jax.jit
def _snake_crashed(snake: Snake) -> jax.Array:
    snake_head = snake[0]
    return (
        jnp.any(snake_head < 0)
        | jnp.any(snake_head >= GRID_SIZE)
        | jnp.any(
            jax.vmap(lambda x: jnp.all(x == snake_head) & jnp.any(x != -1))(snake[1:])
        )
    )


@jax.jit
def _ate_food(snake: Snake, food: Food) -> jax.Array:
    return jnp.all(snake[0] == food)


@jax.jit
def _move_snake(snake: Snake, action: Action) -> Snake:
    head = snake[0] + action

    snake_length = compute_snake_lenght(snake)

    valid_mask = jnp.arange(snake.shape[0])[:, None] < snake_length

    shifted_snake = jnp.where(valid_mask, jnp.roll(snake, 1, axis=0), -1)

    new_snake = shifted_snake.at[0].set(head)

    return new_snake


@jax.jit
def _grow_snake(old_snake: Snake, moved_snake: Snake) -> Snake:
    snake_length = compute_snake_lenght(moved_snake)

    valid_mask = jnp.arange(moved_snake.shape[0])[:, None] < snake_length

    grown_snake = jnp.where(valid_mask, moved_snake, jnp.roll(old_snake, 1, axis=0))

    return grown_snake
