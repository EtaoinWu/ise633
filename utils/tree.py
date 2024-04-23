import beartype
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype.typing import Iterable
import jax
from jaxtyping import PyTree, Shaped, Array, Integer, Scalar, jaxtyped

typechecked = jaxtyped(typechecker=beartype.beartype)

IntLike = int | Integer[Scalar, ""]


class DynamicStatic[T, U](eqx.Module):
    """Trivial module for holding static info in its PyTree."""

    dynamic: T
    static: U = eqx.field(static=True)


@typechecked
def tree_stack(
    trees: Iterable[PyTree[Shaped[Array, "*?moredims"], " TreeStruct"]],
) -> PyTree[Shaped[Array, "n *?moredims"], " TreeStruct"]:
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


@typechecked
def tree_unstack(
    tree: PyTree[Shaped[Array, "n *?moredims"], " TreeStruct"],
) -> list[PyTree[Shaped[Array, "*?moredims"], " TreeStruct"]]:
    leaves, treedef = jtu.tree_flatten(tree)
    return [
        treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)
    ]


@typechecked
def nfold_cross_validation_trains(
    n: IntLike, x: Shaped[Array, " d *moredims"]
) -> Shaped[Array, " {n} d/{n}*{n-1} *moredims"]:
    assert (
        x.shape[0] % n == 0
    ), f"0th dimension={x.shape[0]} must be divisible by {n}"
    # assert 0 <= i < n, f"i={i} must be in [0, {n})"
    slices = jnp.reshape(x, (1, n, -1, *x.shape[1:]))
    duped = jnp.repeat(slices, n, axis=0)
    rolls = jax.vmap(jnp.roll, in_axes=(0, 0, None))(
        duped, -jnp.arange(n), 0
    )
    popped = rolls[:, 1:, ...]
    return jnp.reshape(popped, (n, -1, *x.shape[1:]))


@typechecked
def nfold_cross_validation_tests(
    n: IntLike, x: Shaped[Array, " d *moredims"]
) -> Shaped[Array, " {n} d/{n} *moredims"]:
    assert (
        x.shape[0] % n == 0
    ), f"0th dimension={x.shape[0]} must be divisible by {n}"
    slices = jnp.reshape(x, (n, -1, *x.shape[1:]))
    return slices


@typechecked
def tree_nfold_cross_validation_trains(
    n: IntLike,
    x: PyTree[Shaped[Array, " d *?moredims"], " TreeStruct"],
) -> PyTree[
    Shaped[Array, " {n} d/{n}*{n-1} *?moredims"], " TreeStruct"
]:
    return jtu.tree_map(
        lambda v: nfold_cross_validation_trains(n, v), x
    )


@typechecked
def tree_nfold_cross_validation_tests(
    n: IntLike,
    x: PyTree[Shaped[Array, " d *?moredims"], " TreeStruct"],
) -> PyTree[Shaped[Array, " {n} d/{n} *?moredims"], " TreeStruct"]:
    return jtu.tree_map(lambda v: nfold_cross_validation_tests(n, v), x)
