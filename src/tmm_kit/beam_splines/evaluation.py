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

    polyline: Sequence[Sequence[float]] | np.ndarray
    _tangents: np.ndarray = field(init=False, repr=False)
    _normals: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute geometry shared by element and global evaluation."""
        self.polyline = np.asarray(self.polyline, dtype=np.float64)
        if self.polyline.ndim != 2 or self.polyline.shape[1] != 2:
            raise ValueError("The array points must be 2D vectors")
        if self.polyline.shape[0] < 2:
            raise ValueError("The number of array points must be greater or equal to 2")

        element_v = np.diff(self.polyline, axis=0)
        element_l = np.linalg.norm(element_v, axis=1)
        self._tangents = element_v / element_l[:, None]
        self._normals = np.column_stack((-self._tangents[:, 1], self._tangents[:, 0]))

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

    def polyline_for_dummies(
        self,
        dummy_nodes_per_element: int | Sequence[int] | np.ndarray = 0,
    ) -> np.ndarray:
        """Return a polyline prepared for inserting dummy nodes.

        Parameters
        ----------
        dummy_nodes_per_element
            Number of dummy nodes inserted into each element:
            a scalar applies the same number to every element;
            an array specifies a number for each element.
        """
        elements_n = len(self.tmm_result.elements)

        # Process dummy nodes number
        dummy_numbers_arr = np.asarray(dummy_nodes_per_element, dtype=int)
        if dummy_numbers_arr.size == 1:
            dummy_numbers_arr = np.repeat(dummy_numbers_arr.item(), elements_n)
        elif dummy_numbers_arr.size != elements_n:
            raise ValueError(
                f"Length of dummy_nodes_per_element ({dummy_numbers_arr.size}) "
                f"must match the number of elements ({elements_n})."
            )

        # Instantiate a new polyline
        dummy_nodes_n = sum(dummy_numbers_arr)
        new_points_n = len(self.polyline) + dummy_nodes_n
        new_polyline = np.zeros(
            (new_points_n, self.polyline.shape[1]), dtype=np.float64
        )

        new_point_i = 0
        for i, real_point in enumerate(self.polyline):
            # Insert a real point
            new_polyline[new_point_i] = real_point
            new_point_i += 1

            if new_point_i == new_points_n:
                break

            # Insert dummy points
            dummy_n = dummy_numbers_arr[i]

            el_l = self.tmm_result.elements[i].length
            s = np.linspace(0, el_l, dummy_n + 2)[1:-1]
            new_polyline[new_point_i : new_point_i + dummy_n] = (
                self.evaluate_element_point(i, s)
            )

            new_point_i += dummy_n

        return new_polyline
