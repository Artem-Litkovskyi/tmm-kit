from collections.abc import Sequence

import numpy as np


def adaptive_stiffness(
    element_length: float | np.ndarray,
    bandwidth: float,
):
    """Calculate adaptive beam support stiffness for step-independent smoothing.

    Parameters
    ----------
    element_length
        Array of element lengths. For better results,
        calculate using centered_element_length.
    bandwidth
        Smoothing bandwidth
    """
    return element_length / bandwidth**4


def smoothing_spline_response(t_divider, bandwidth):
    """Calculate smoothing spline response.

    Notes
    -----
    It shows how much a smoothing spline of a given bandwidth
    attenuates a y(t) = cos(t / t_divider) signal.
    """
    return t_divider**4 / (bandwidth**4 + t_divider**4)


def centered_element_length(
    element_length: Sequence[float] | np.ndarray,
    loop: bool = False,
):
    """Calculate a mean of the previous and the next element lengths per point.

    Parameters
    ----------
    element_length
        Array of element lengths
    loop
        Set to True if the last element's end connects to the start of the first one
    """
    length_arr = np.asarray(element_length, dtype=np.float64)
    if loop:
        a = length_arr[-1]
        b = length_arr[0]
    else:
        a = length_arr[0]
        b = length_arr[-1]
    return (np.concat(([a], length_arr)) + np.concat((length_arr, [b]))) / 2.0
