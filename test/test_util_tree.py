import pytest
from jax import numpy as jnp, tree_util as jtu
from utils.tree import (
    tree_stack,
    tree_unstack,
    nfold_cross_validation_trains,
    tree_nfold_cross_validation_trains,
)
from chex import assert_trees_all_equal


def test_tree_stack():
    trees = [
        {"x": jnp.array([1, 2, 3]), "y": jnp.array([4, 5, 6])},
        {"x": jnp.array([7, 8, 9]), "y": jnp.array([10, 11, 12])},
    ]
    expected_result = {
        "x": jnp.array([[1, 2, 3], [7, 8, 9]]),
        "y": jnp.array([[4, 5, 6], [10, 11, 12]]),
    }
    assert_trees_all_equal(tree_stack(trees), expected_result)
    trees = [
        (jnp.array([1, 2]), [jnp.array([3, 4]), jnp.array([5, 6])]),
        (jnp.array([7, 8]), [jnp.array([9, 10]), jnp.array([11, 12])]),
    ]
    expected_result = (
        jnp.array([[1, 2], [7, 8]]),
        [jnp.array([[3, 4], [9, 10]]), jnp.array([[5, 6], [11, 12]])],
    )
    assert_trees_all_equal(tree_stack(trees), expected_result)
    trees = [
        (jnp.array([1, 2]), jnp.array([[3, 4, 5], [6, 7, 8]])),
        (jnp.array([9, 10]), jnp.array([[11, 12, 13], [14, 15, 16]])),
    ]
    expected_result = (
        jnp.array([[1, 2], [9, 10]]),
        jnp.array(
            [[[3, 4, 5], [6, 7, 8]], [[11, 12, 13], [14, 15, 16]]]
        ),
    )
    assert_trees_all_equal(tree_stack(trees), expected_result)


def test_tree_unstack():
    tree = {
        "x": jnp.array([[1, 2, 3], [7, 8, 9]]),
        "y": jnp.array([[4, 5, 6], [10, 11, 12]]),
    }
    expected_result = [
        {"x": jnp.array([1, 2, 3]), "y": jnp.array([4, 5, 6])},
        {"x": jnp.array([7, 8, 9]), "y": jnp.array([10, 11, 12])},
    ]
    assert_trees_all_equal(tree_unstack(tree), expected_result)
    tree = (
        jnp.array([[1, 2], [7, 8]]),
        [jnp.array([[3, 4], [9, 10]]), jnp.array([[5, 6], [11, 12]])],
    )
    expected_result = [
        (jnp.array([1, 2]), [jnp.array([3, 4]), jnp.array([5, 6])]),
        (jnp.array([7, 8]), [jnp.array([9, 10]), jnp.array([11, 12])]),
    ]
    assert_trees_all_equal(tree_unstack(tree), expected_result)
    tree = (
        jnp.array([[1, 2], [9, 10]]),
        jnp.array(
            [[[3, 4, 5], [6, 7, 8]], [[11, 12, 13], [14, 15, 16]]]
        ),
    )
    expected_result = [
        (jnp.array([1, 2]), jnp.array([[3, 4, 5], [6, 7, 8]])),
        (jnp.array([9, 10]), jnp.array([[11, 12, 13], [14, 15, 16]])),
    ]
    assert_trees_all_equal(tree_unstack(tree), expected_result)


def test_nfold_cross_validation():
    data = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    assert_trees_all_equal(
        nfold_cross_validation_trains(2, data),
        jnp.array([[[7, 8, 9], [10, 11, 12]], [[1, 2, 3], [4, 5, 6]]]),
    )
    trains_4 = nfold_cross_validation_trains(4, data)
    assert_trees_all_equal(
        trains_4[0],
        jnp.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]]),
    )
    assert_trees_all_equal(
        trains_4[1],
        jnp.array([[7, 8, 9], [10, 11, 12], [1, 2, 3]]),
    )


def test_tree_nfold_cross_validation():
    data = {
        "x": jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        "y": jnp.array([13, 14, 15, 16]),
    }
    trains_2 = tree_nfold_cross_validation_trains(2, data)
    assert_trees_all_equal(
        {
            "x": jnp.array([[7, 8, 9], [10, 11, 12]]),
            "y": jnp.array([15, 16]),
        },
        jtu.tree_map(lambda v: v[0], trains_2),
    )
    trains_4 = tree_nfold_cross_validation_trains(4, data)
    assert_trees_all_equal(
        jtu.tree_map(lambda v: v[1], trains_4),
        {
            "x": jnp.array([[7, 8, 9], [10, 11, 12], [1, 2, 3]]),
            "y": jnp.array([15, 16, 13]),
        },
    )
