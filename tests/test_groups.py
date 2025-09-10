import jax.numpy as jnp

from src import groups

D4_rep = groups.make_D4_permutation_representation()
identity = jnp.arange(4)


def test_compose_permutations():
    rot90 = jnp.array([3, 2, 0, 1])
    flip_updown = jnp.array([0, 1, 3, 2])
    flip_topleft_downright = jnp.array([3, 2, 1, 0])
    flip_topright_downleft = jnp.array([2, 3, 0, 1])

    assert (
        groups.compose_permutation(rot90, flip_updown) == flip_topright_downleft
    ).all()

    assert (
        groups.compose_permutation(flip_updown, rot90) == flip_topleft_downright
    ).all()


def test_inverse_permutation():
    for tsf_name, (_, perm) in D4_rep.items():
        assert (
            groups.compose_permutation(groups.inverse_perm(perm), perm) == identity
        ).all()
        assert (
            groups.compose_permutation(perm, groups.inverse_perm(perm)) == identity
        ).all()


def test_D4_permutations():
    assert (D4_rep["rot0"][1] == jnp.array([0, 1, 2, 3])).all()
    assert (D4_rep["rot90"][1] == jnp.array([3, 2, 0, 1])).all()
    assert (D4_rep["rot180"][1] == jnp.array([1, 0, 3, 2])).all()
    assert (D4_rep["rot270"][1] == jnp.array([2, 3, 1, 0])).all()

    assert (D4_rep["flip_rot0"][1] == jnp.array([0, 1, 3, 2])).all()
    assert (D4_rep["flip_rot90"][1] == jnp.array([2, 3, 0, 1])).all()
    assert (D4_rep["flip_rot180"][1] == jnp.array([1, 0, 2, 3])).all()
    assert (D4_rep["flip_rot270"][1] == jnp.array([3, 2, 1, 0])).all()


def test_group_multiplication_table():
    assert (
        groups.compose_permutation(D4_rep["rot90"][1], D4_rep["rot270"][1]) == identity
    ).all()

    assert (
        groups.compose_permutation(D4_rep["rot180"][1], D4_rep["rot180"][1]) == identity
    ).all()

    assert (
        groups.compose_permutation(D4_rep["rot90"][1], D4_rep["flip_rot0"][1])
        == D4_rep["flip_rot90"][1]
    ).all()

    assert (
        groups.compose_permutation(D4_rep["rot180"][1], D4_rep["flip_rot0"][1])
        == D4_rep["flip_rot180"][1]
    ).all()

    assert (
        groups.compose_permutation(D4_rep["flip_rot0"][1], D4_rep["rot270"][1])
        == D4_rep["flip_rot90"][1]
    ).all()

    assert (
        groups.compose_permutation(D4_rep["flip_rot0"][1], D4_rep["rot180"][1])
        == D4_rep["flip_rot180"][1]
    ).all()

    assert (
        groups.compose_permutation(D4_rep["rot180"][1], D4_rep["flip_rot0"][1])
        == D4_rep["flip_rot180"][1]
    ).all()
