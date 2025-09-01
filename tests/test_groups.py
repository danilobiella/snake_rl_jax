import jax.numpy as jnp

from src import groups


def test_compose_permutations():
    rot90 = jnp.array([2, 3, 1, 0])
    flip_updown = jnp.array([0, 1, 3, 2])
    flip_topright_downleft = jnp.array([2, 3, 0, 1])
    flip_topleft_downright = jnp.array([3, 2, 1, 0])

    assert (
        groups.compose_permutation(rot90, flip_updown) == flip_topright_downleft
    ).all()

    assert (
        groups.compose_permutation(flip_updown, rot90) == flip_topleft_downright
    ).all()


def test_D4_permutations():
    D4_permutation_rep = groups.make_D4_permutation_representation()
    D4_permutations = {k: v[1] for k, v in D4_permutation_rep.items()}

    assert (D4_permutations["rot0"] == jnp.array([0, 1, 2, 3])).all()
    assert (D4_permutations["rot90"] == jnp.array([2, 3, 1, 0])).all()
    assert (D4_permutations["rot180"] == jnp.array([1, 0, 3, 2])).all()
    assert (D4_permutations["rot270"] == jnp.array([3, 2, 0, 1])).all()
    assert (D4_permutations["rot0_flip"] == jnp.array([0, 1, 3, 2])).all()
    assert (D4_permutations["rot90_flip"] == jnp.array([2, 3, 0, 1])).all()
    assert (D4_permutations["rot180_flip"] == jnp.array([1, 0, 2, 3])).all()
    assert (D4_permutations["rot270_flip"] == jnp.array([3, 2, 1, 0])).all()
