import numpy as np

from tmm_kit.utils import subdivide_array

NUMBERS = np.array([0, 1, 3], dtype=np.float64)

VECTORS = np.array(
    [
        [0, 0],
        [2, 4],
        [4, 3],
    ],
    dtype=np.float64,
)


def test_subdivide_array_numbers_subdivisions_const():
    b, indices = subdivide_array(NUMBERS, 2)

    np.testing.assert_array_equal(indices, [0, 0, 1, 1, 1])
    np.testing.assert_allclose(
        b,
        [0, 0.5, 1, 2, 3],
    )


def test_subdivide_array_numbers_subdivisions_array():
    b, indices = subdivide_array(NUMBERS, [1, 2])

    np.testing.assert_array_equal(indices, [0, 1, 1, 1])
    np.testing.assert_allclose(
        b,
        [0, 1, 2, 3],
    )


def test_subdivide_array_vectors_subdivisions_const():
    b, indices = subdivide_array(VECTORS, 2)

    np.testing.assert_array_equal(indices, [0, 0, 1, 1, 1])
    np.testing.assert_allclose(
        b,
        [
            [0, 0],
            [1, 2],
            [2, 4],
            [3, 3.5],
            [4, 3],
        ],
    )


def test_subdivide_array_vectors_subdivisions_array():
    b, indices = subdivide_array(VECTORS, [1, 2])

    np.testing.assert_array_equal(indices, [0, 1, 1, 1])
    np.testing.assert_allclose(
        b,
        [
            [0, 0],
            [2, 4],
            [3, 3.5],
            [4, 3],
        ],
    )
