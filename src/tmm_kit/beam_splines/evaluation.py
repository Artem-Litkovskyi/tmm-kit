from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

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
    support_t: Sequence[float] | np.ndarray

    def evaluate_element_state(
        self, element_i: int, s: np.typing.ArrayLike
    ) -> np.ndarray:
        """Evaluate a beam state at a given element coordinate."""
        return self.tmm_result.evaluate(element_i, s)

    def evaluate_spline_state(self, t: np.typing.ArrayLike) -> np.ndarray:
        """Evaluate a beam state at a given spline coordinate."""
        return self._evaluate_spline(
            t,
            (self.tmm_result.state_size,),
            self.evaluate_element_state,
        )

    def _evaluate_spline(
        self,
        t: np.typing.ArrayLike,
        result_shape: tuple[int, ...],
        evaluate_element: Callable[[int, np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Evaluate an element-wise quantity at spline coordinates."""
        eval_ts = np.atleast_1d(np.asarray(t, dtype=np.float64))
        result = np.empty((eval_ts.size, *result_shape), dtype=np.float64)

        element_indices = np.searchsorted(self.support_t, eval_ts, side="right") - 1
        element_indices = np.clip(element_indices, 0, len(self.support_t) - 2)

        for element_i in np.unique(element_indices):
            state_indices = element_indices == element_i
            s = eval_ts[state_indices] - self.support_t[element_i]
            result[state_indices] = evaluate_element(int(element_i), s)

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
        support_t: Sequence[float] | np.ndarray,
        polyline: Sequence[Sequence[float]] | np.ndarray,
    ) -> None:
        super().__init__(tmm_result, support_t)

        polyline_arr = np.asarray(polyline, dtype=np.float64)
        if polyline_arr.ndim != 2 or polyline_arr.shape[1] != 2:
            raise ValueError("The array points must be 2D vectors")
        if polyline_arr.shape[0] < 2:
            raise ValueError("The number of array points must be greater or equal to 2")

        self.polyline = polyline_arr

        element_v = np.diff(polyline_arr, axis=0)
        element_l = np.linalg.norm(element_v, axis=1)
        self._tangents = element_v / element_l[:, None]
        self._normals = np.column_stack(
            (-self._tangents[:, 1], self._tangents[:, 0])
        )

    def evaluate_element_point(
        self, element_i: int, s: np.typing.ArrayLike
    ) -> np.ndarray:
        """Evaluate a point at a given element coordinate."""
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        states = self.tmm_result.evaluate(element_i, s_arr)
        W, theta, _, _ = states.T

        local_x = s_arr - W * np.sin(theta)
        local_y = W * np.cos(theta)

        return (
            self.polyline[element_i]
            + np.outer(local_x, self._tangents[element_i])
            + np.outer(local_y, self._normals[element_i])
        )

    def evaluate_element_curvature(
        self, element_i: int, s: np.typing.ArrayLike
    ) -> np.ndarray:
        """Evaluate a curvature at a given element coordinate."""
        s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
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

        numerator = d2x_ds2 * dy_ds - d2y_ds2 * dx_ds
        denominator = np.pow(np.square(dx_ds) + np.square(dy_ds), 3 / 2)

        return numerator / denominator

    def evaluate_spline_point(self, t: np.typing.ArrayLike) -> np.ndarray:
        """Evaluate a point at a given spline coordinate."""
        return self._evaluate_spline(
            t,
            (2,),
            self.evaluate_element_point,
        )

    def evaluate_spline_curvature(self, t: np.typing.ArrayLike) -> np.ndarray:
        """Evaluate a curvature at a given spline coordinate."""
        return self._evaluate_spline(
            t,
            (),
            self.evaluate_element_curvature,
        )
