from typing import Callable

import jax.numpy as jnp


def identity(x):
    return x


def compose_permutation(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """
    Compose permutations. First it applies q and then p
    """
    return p[q]


def compose_transformation(f: Callable, g: Callable) -> Callable:
    """
    Compose functions. First it applies g and then f
    """
    return lambda x: f(g(x))


def rot_90(x):
    return jnp.rot90(x, k=1, axes=(0, 1))


def flip_updown(x):
    return jnp.flip(x, axis=1)


def make_D4_permutation_representation():
    flip_updown_perm = jnp.array([0, 1, 3, 2])  # R L U D
    r90_perm = jnp.array([2, 3, 1, 0])  # D U L R

    D4_permutation_representation = {}
    transformation = identity
    perm = jnp.array(range(4))
    for s in range(2):
        for k in range(4):
            D4_permutation_representation[f"rot{90*k}" + "_flip" * s] = (
                transformation,
                perm,
            )

            transformation = compose_transformation(rot_90, transformation)
            perm = compose_permutation(r90_perm, perm)

        transformation = compose_transformation(flip_updown, transformation)
        perm = compose_permutation(flip_updown_perm, perm)

    return D4_permutation_representation
