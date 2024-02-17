import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype.typing import Iterable


class DynamicStatic[T, U](eqx.Module):
    """Trivial module for holding static info in its PyTree."""

    dynamic: T
    static: U = eqx.field(static=True)


def tree_stack[T](trees: Iterable[T]) -> T:
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_unstack[T](tree: T) -> list[T]:
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
