import numpy as np


def subdivide_array(
    array: np.ndarray,
    subdivisions: int | np.typing.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly subdivide consecutive entries of an array.

    Parameters
    ----------
    array
        Ordered entries to subdivide. A one-dimensional array represents
        scalar values; a two-dimensional array with shape ``(N, D)``
        represents ``N`` points in ``D``-dimensional space. At least two
        entries are required.
    subdivisions
        Number of samples to generate for each segment. A scalar applies to
        every segment; an array-like value must contain exactly ``N - 1``
        values. Every value must be at least one.
    """
    elements_n = array.shape[0] - 1

    subs = np.asarray(subdivisions, dtype=int)
    if subs.size == 1:
        subs = np.repeat(subs.item(), elements_n)
    elif subs.size != elements_n:
        raise ValueError(
            f"Length of subdivisions ({subs.size}) "
            f"must match number of elements ({elements_n})."
        )

    if np.any(subs < 1):
        raise ValueError("Number of subdivisions must be at least 1.")

    new_items = []
    element_indices = []

    for i in range(elements_n):
        start = array[i]
        end = array[i + 1]

        t = np.linspace(0.0, 1.0, subs[i], endpoint=False)

        seg_points = start + np.outer(t, (end - start))

        new_items.append(seg_points)
        element_indices.append(np.repeat(i, subs[i]))

    new_items.append(array[-1].reshape(1, -1))
    element_indices.append([elements_n - 1])

    new_items = np.vstack(new_items)
    element_indices = np.concatenate(element_indices)

    if array.ndim == 1:
        new_items = new_items.ravel()

    return new_items, element_indices
