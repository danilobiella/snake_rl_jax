from typing import Callable

import jax.numpy as jnp


def identity(x):
    return x


def compose_permutation(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    return p[q]


def compose_transformation(f: Callable, g: Callable) -> Callable:
    return lambda x: f(g(x))


def rot_90(x):
    return jnp.rot90(x, k=1, axes=(0, 1))


def vertical_reflection(x):
    return jnp.flip(x, axis=1)


def board_to_emoji(arr: jnp.ndarray) -> str:
    """
    Convert a 2D array board into an emoji string.
    Convention:
      -1 → 🍎 (apple / food)
       0 → ⬛ (empty)
       1 → 🟩 (snake head/body)
    """
    mapping = {
        -1: "🍎",
        0: "⬛",
        1: "🟩",
    }
    lines = []
    for row in arr:
        line = "".join(mapping.get(int(v), "❓") for v in row)
        lines.append(line)
    return "\n".join(lines)


def make_D4_permutation_representation():
    r90_perm = jnp.array([2, 3, 1, 0])  # D U L R
    vertical_reflection_perm = jnp.array([0, 1, 3, 2])  # R L U D

    D4_permutation_representation = {}
    transformation = identity
    perm = jnp.array(range(4))
    for s in range(2):
        for k in range(4):
            D4_permutation_representation[f"rot{90*k}" + "_reflection" * s] = (
                transformation,
                perm,
            )

            transformation = compose_transformation(rot_90, transformation)
            perm = compose_permutation(r90_perm, perm)

        transformation = compose_transformation(vertical_reflection, transformation)
        perm = compose_permutation(vertical_reflection_perm, perm)

    return D4_permutation_representation
