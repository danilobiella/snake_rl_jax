from typing import Callable

import jax.numpy as jnp


def identity(x):
    return x


def compose_permutation(p: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """
    Compose permutations. First it applies q and then p
    """
    return q[p]


def compose_transformation(f: Callable, g: Callable) -> Callable:
    """
    Compose functions. First it applies g and then f
    """
    return lambda x: f(g(x))


def inverse_perm(perm: jnp.ndarray) -> jnp.ndarray:
    return jnp.argsort(perm)


def rot_90(x):
    return jnp.rot90(x, k=-1, axes=(0, 1))


def flip_updown(x):
    return jnp.flip(x, axis=0)


identity_perm = jnp.array([0, 1, 2, 3])  # R L D U
r90_perm = jnp.array([3, 2, 0, 1])  # U D R L
flip_updown_perm = jnp.array([0, 1, 3, 2])  # R L U D


def make_C4_permutation_representation():
    C4_rep = {}
    transformation = identity
    perm = jnp.array(range(4))
    for k in range(4):
        C4_rep[f"rot{90*k}"] = (
            transformation,
            perm,
        )

        transformation = compose_transformation(rot_90, transformation)
        perm = compose_permutation(r90_perm, perm)

    return C4_rep


def make_D4_permutation_representation():
    D4_rep = {}
    transformation = identity
    perm = identity_perm
    for s in range(2):
        for k in range(4):
            D4_rep["flip_" * s + f"rot{90*k}"] = (
                transformation,
                perm,
            )

            transformation = compose_transformation(rot_90, transformation)
            perm = compose_permutation(r90_perm, perm)

        transformation = compose_transformation(flip_updown, transformation)
        perm = compose_permutation(flip_updown_perm, perm)

    return D4_rep
