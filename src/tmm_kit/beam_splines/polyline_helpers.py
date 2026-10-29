from collections.abc import Sequence

import numpy as np

from tmm_kit.beam_splines import CorotationalBeamSpline


def angle_misalignment(polyline: Sequence[Sequence[float]] | np.ndarray):
    """Get angle misalignment in radians for every array point.

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
    new_points_per_segment: int | Sequence[int] | np.ndarray = 0,
) -> np.ndarray:
    """Create a denser polyline whose added points lie on the spline.

    Parameters
    ----------
    spline
        Corotational beam spline, the polyline of which needs to be refined.
    new_points_per_segment
        Number of points inserted into each polyline segment:
        a scalar applies the same number to every segment;
        an array specifies a number for each segment.
    """
    segments_n = len(spline.polyline) - 1

    # Process added points number
    dummy_numbers_arr = np.asarray(new_points_per_segment, dtype=int)
    if dummy_numbers_arr.size == 1:
        dummy_numbers_arr = np.repeat(dummy_numbers_arr.item(), segments_n)
    elif dummy_numbers_arr.size != segments_n:
        raise ValueError(
            f"Length of new_points_per_segment ({dummy_numbers_arr.size}) "
            f"must match the number of segments ({segments_n})."
        )

    # Instantiate a new polyline
    dummy_nodes_n = sum(dummy_numbers_arr)
    new_points_n = len(spline.polyline) + dummy_nodes_n
    new_polyline = np.zeros(
        (new_points_n, spline.polyline.shape[1]), dtype=np.float64
    )

    new_point_i = 0
    for i, supported_point in enumerate(spline.polyline):
        # Insert a supported point
        new_polyline[new_point_i] = supported_point
        new_point_i += 1

        if new_point_i == new_points_n:
            break

        # Insert dummy points
        dummy_n = dummy_numbers_arr[i]

        el_l = spline.tmm_result.elements[i].length
        s = np.linspace(0, el_l, dummy_n + 2)[1:-1]
        new_polyline[new_point_i : new_point_i + dummy_n] = (
            spline.evaluate_element_point(i, s)
        )

        new_point_i += dummy_n

    return new_polyline
