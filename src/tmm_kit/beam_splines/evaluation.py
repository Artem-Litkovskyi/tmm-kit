from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from tmm_kit import ElementEdge
from tmm_kit.tmm.result import TMMResult


@dataclass
class BeamSpline:
    """Beam spline evaluator.

    Attributes
    ----------
    tmm_result
        Result of TMM solver
    support_t
        Parameter t coordinates at beam supports.
    """

    tmm_result: TMMResult
    support_t: npt.ArrayLike
    _support_t_arr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize support_t into a float array."""
        self._support_t_arr = np.asarray(self.support_t, dtype=np.float64)

    @property
    def element_lengths(self) -> np.ndarray:
        """Get element lengths array."""
        return np.array(
            [element.length for element in self.tmm_result.elements],
            dtype=np.float64,
        )

    def evaluate_element_state(
        self, element_i: npt.ArrayLike, s: npt.ArrayLike
    ) -> np.ndarray:
        """Evaluate a beam state at a given element coordinate."""
        return self.tmm_result.evaluate(element_i, s)

    def evaluate_spline_state(self, t: npt.ArrayLike) -> np.ndarray:
        """Evaluate a beam state at a given spline coordinate."""
        return self._evaluate_spline(t, self.evaluate_element_state)

    def get_edge_state(
        self,
        element_i: npt.ArrayLike,
        element_edge: ElementEdge | None = None,
    ) -> np.ndarray:
        """Get state vectors from the array of all element states."""
        return self.tmm_result.get_edge_state(element_i, element_edge)

    def get_all_edge_states(
        self,
        element_edge: ElementEdge | None = None,
    ) -> np.ndarray:
        """Get all state vectors from the array of all element states."""
        return self.tmm_result.get_all_edge_states(element_edge)

    def _evaluate_spline(
        self,
        t: npt.ArrayLike,
        evaluate_element: Callable[..., np.ndarray],
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate an element-wise quantity at spline coordinates.

        The evaluate_element is called once with the arrays of element indices
        and local coordinates. If states are given, they are its third argument.
        """
        eval_ts = np.atleast_1d(np.asarray(t, dtype=np.float64))

        element_indices = (
            np.searchsorted(self._support_t_arr, eval_ts, side="right") - 1
        )
        element_indices = np.clip(element_indices, 0, self._support_t_arr.size - 2)
        s = eval_ts - self._support_t_arr[element_indices]

        if states is None:
            result = evaluate_element(element_indices, s)
        else:
            result = evaluate_element(element_indices, s, np.atleast_2d(states))

        if np.isscalar(t):
            return result[0]
        return result


@dataclass
class CorotationalBeamSpline(BeamSpline):
    """Corotational beam spline evaluator.

    Attributes
    ----------
    tmm_result
        Result of TMM solver
    support_t
        Parameter t coordinates at polyline points.
    polyline
        Coordinates of the array points.
    """

    polyline: np.ndarray
    _tangents: np.ndarray = field(init=False, repr=False)
    _normals: np.ndarray = field(init=False, repr=False)

    def __init__(
        self,
        tmm_result: TMMResult,
        support_t: npt.ArrayLike,
        polyline: npt.ArrayLike,
    ) -> None:
        super().__init__(tmm_result, support_t)

        polyline_arr = np.asarray(polyline, dtype=np.float64)
        if polyline_arr.ndim != 2 or polyline_arr.shape[1] != 2:
            raise ValueError("The array points must be 2D vectors")
        if polyline_arr.shape[0] < 2:
            raise ValueError("The number of array points must be greater or equal to 2")

        self.polyline = polyline_arr

        element_v = np.diff(polyline_arr, axis=0)
        self._tangents = element_v / self.element_lengths[:, None]
        self._normals = np.column_stack((-self._tangents[:, 1], self._tangents[:, 0]))

    def evaluate_element_normal(
        self,
        element_i: npt.ArrayLike,
        s: npt.ArrayLike,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate a rotated normal at a given element coordinate.

        Parameters
        ----------
        element_i
            Element index, or an array of indices, one per coordinate.
        s
            Local coordinate (coordinates) from 0 to the element length.
        states
            Precomputed states at the same coordinates, shape (N, 4).
            Evaluated if None.
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        if states is None:
            states = self.tmm_result.evaluate(element_i, s_arr)
        _, theta, _, _ = states.T

        local_x = -np.sin(theta)
        local_y = np.cos(theta)

        return (
            local_x[:, None] * self._tangents[element_i]
            + local_y[:, None] * self._normals[element_i]
        )

    def evaluate_element_point(
        self,
        element_i: npt.ArrayLike,
        s: npt.ArrayLike,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate a point at a given element coordinate.

        Parameters
        ----------
        element_i
            Element index, or an array of indices, one per coordinate.
        s
            Local coordinate (coordinates) from 0 to the element length.
        states
            Optional precomputed states at the same coordinates, shape (N, 4).
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        if states is None:
            states = self.tmm_result.evaluate(element_i, s_arr)
        W, theta, _, _ = states.T

        local_x = s_arr - W * np.sin(theta)
        local_y = W * np.cos(theta)

        return (
            self.polyline[element_i]
            + local_x[:, None] * self._tangents[element_i]
            + local_y[:, None] * self._normals[element_i]
        )

    def evaluate_element_curvature(
        self,
        element_i: npt.ArrayLike,
        s: npt.ArrayLike,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate a curvature at a given element coordinate.

        Parameters
        ----------
        element_i
            Element index, or an array of indices, one per coordinate.
        s
            Local coordinate (coordinates) from 0 to the element length.
        states
            Optional precomputed states at the same coordinates, shape (N, 4).
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        if states is None:
            states = self.tmm_result.evaluate(element_i, s_arr)
        W, theta, M, Q = states.T

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        W_M = W * M
        two_theta_M_plus_W_Q = 2.0 * theta * M + W * Q
        M_minus_W_M2 = M - W * np.square(M)

        dx_ds = 1.0 - theta * sin_theta - W_M * cos_theta
        dy_ds = theta * cos_theta - W_M * sin_theta

        d2x_ds2 = M_minus_W_M2 * sin_theta + two_theta_M_plus_W_Q * cos_theta
        d2y_ds2 = M_minus_W_M2 * cos_theta - two_theta_M_plus_W_Q * sin_theta

        numerator = dx_ds * d2y_ds2 - dy_ds * d2x_ds2
        denominator = np.pow(np.square(dx_ds) + np.square(dy_ds), 3 / 2)

        return numerator / denominator

    def evaluate_spline_normal(
        self,
        t: npt.ArrayLike,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate a rotated normal at a given spline coordinate.

        Parameters
        ----------
        t
            Spline coordinate (coordinates).
        states
            Optional precomputed states at the same coordinates, shape (N, 4).
        """
        return self._evaluate_spline(t, self.evaluate_element_normal, states)

    def evaluate_spline_point(
        self,
        t: npt.ArrayLike,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate a point at a given spline coordinate.

        Parameters
        ----------
        t
            Spline coordinate (coordinates).
        states
            Optional precomputed states at the same coordinates, shape (N, 4).
        """
        return self._evaluate_spline(t, self.evaluate_element_point, states)

    def evaluate_spline_curvature(
        self,
        t: npt.ArrayLike,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate a curvature at a given spline coordinate.

        Parameters
        ----------
        t
            Spline coordinate (coordinates).
        states
            Optional precomputed states at the same coordinates, shape (N, 4).
        """
        return self._evaluate_spline(t, self.evaluate_element_curvature, states)
