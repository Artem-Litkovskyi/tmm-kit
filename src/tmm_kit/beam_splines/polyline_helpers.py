from collections.abc import Sequence

import numpy as np

from tmm_kit.beam_splines import CorotationalBeamSpline


def angle_misalignment(polyline: Sequence[Sequence[float]] | np.ndarray):
    """Calculate angle misalignment in radians for every array point.

    Notes
    -----
    Used as an argument for conjugation equations of corotational beam splines.
    """
    polyline_arr = np.asarray(polyline, dtype=np.float64)

    segment_v = np.diff(polyline_arr, axis=0)

    prev_v = np.concatenate((segment_v[-1:], segment_v))
    next_v = np.concatenate((segment_v, segment_v[:1]))

    dot = (prev_v * next_v).sum(axis=1)
    cross = next_v[:, 0] * prev_v[:, 1] - next_v[:, 1] * prev_v[:, 0]

    return np.arctan2(cross, dot)


def refine_polyline(
    spline: CorotationalBeamSpline,
    dummy_points_per_segment: int | Sequence[int] | np.ndarray = 0,
) -> np.ndarray:
    """Create a denser polyline whose added points lie on the spline.

    Parameters
    ----------
    spline
        Corotational beam spline, the polyline of which needs to be refined.
    dummy_points_per_segment
        Number of points inserted into each polyline segment:
        a scalar applies the same number to every segment;
        an array specifies a number for each segment.
    """
    element_lengths = spline.element_lengths

    real_points_n = len(spline.polyline)
    segments_n = real_points_n - 1

    dummy_numbers_arr = np.asarray(dummy_points_per_segment, dtype=int)
    if dummy_numbers_arr.size == 1:
        dummy_numbers_arr = np.repeat(dummy_numbers_arr.item(), segments_n)

    real_mask = real_points_mask(real_points_n, dummy_numbers_arr)
    real_indices = np.flatnonzero(real_mask)

    new_polyline = np.zeros(
        (real_mask.size, spline.polyline.shape[1]), dtype=np.float64
    )
    new_polyline[real_mask] = spline.polyline

    for i, dummy_n in enumerate(dummy_numbers_arr):
        if dummy_n == 0:
            continue

        s = np.linspace(0.0, element_lengths[i], dummy_n + 2)[1:-1]
        dummy_indices = np.arange(
            real_indices[i] + 1,
            real_indices[i + 1],
        )

        new_polyline[dummy_indices] = spline.evaluate_element_point(i, s)

    return new_polyline


def real_points_mask(
    real_points_n: int,
    dummy_points_per_segment: int | Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Create a boolean mask for selecting real points in a refined polyline.

    Parameters
    ----------
    real_points_n
        Number of real points in the polyline.
    dummy_points_per_segment
        Number of dummy points inserted into each segment:
        a scalar means that every segment has the same number;
        an array specifies a number for each segment.
    """
    if real_points_n < 2:
        raise ValueError("The number of real points must be at least 2.")

    segments_n = real_points_n - 1

    dummy_numbers_arr = np.asarray(dummy_points_per_segment, dtype=int)
    if dummy_numbers_arr.size == 1:
        dummy_numbers_arr = np.repeat(dummy_numbers_arr.item(), segments_n)
    elif dummy_numbers_arr.size != segments_n:
        raise ValueError(
            f"Length of dummy_points_per_segment ({dummy_numbers_arr.size}) "
            f"must match the number of segments ({segments_n})."
        )

    if np.any(dummy_numbers_arr < 0):
        raise ValueError("The number of dummy points must be non-negative.")

    mask = np.concatenate(
        [
            np.concatenate(([True], np.zeros(dummy_n, dtype=bool)))
            for dummy_n in dummy_numbers_arr
        ]
        + [np.array([True])]
    )

    return mask
