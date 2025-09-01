import jax.numpy as jnp

from src import groups


def test_D4_permutations():
    D4_permutation_rep = groups.make_D4_permutation_representation()
    D4_permutations = {k: v[1] for k, v in D4_permutation_rep.items()}

    assert all(jnp.isclose(D4_permutations["rot0"], jnp.array([0, 1, 2, 3]), rtol=1e-5))
    assert all(
        jnp.isclose(D4_permutations["rot90"], jnp.array([2, 3, 1, 0]), rtol=1e-5)
    )
    assert all(
        jnp.isclose(D4_permutations["rot180"], jnp.array([1, 0, 3, 2]), rtol=1e-5)
    )
    assert all(
        jnp.isclose(D4_permutations["rot270"], jnp.array([3, 2, 0, 1]), rtol=1e-5)
    )
    assert all(
        jnp.isclose(
            D4_permutations["rot0_reflection"], jnp.array([0, 1, 3, 2]), rtol=1e-5
        )
    )
    assert all(
        jnp.isclose(
            D4_permutations["rot90_reflection"], jnp.array([2, 3, 0, 1]), rtol=1e-5
        )
    )
    assert all(
        jnp.isclose(
            D4_permutations["rot180_reflection"], jnp.array([1, 0, 2, 3]), rtol=1e-5
        )
    )
    assert all(
        jnp.isclose(
            D4_permutations["rot270_reflection"], jnp.array([3, 2, 1, 0]), rtol=1e-5
        )
    )
