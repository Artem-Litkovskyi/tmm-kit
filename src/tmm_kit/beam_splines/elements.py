from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from tmm_kit.tmm.base import TMMElement, stack_transfer_eq


@dataclass(slots=True)
class BeamElement(TMMElement):
    """Standard beam spline segment."""

    vectorized: ClassVar[bool] = True

    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f."""
        s_arr = np.asarray(s, dtype=np.float64)
        one, zero = np.ones_like(s_arr), np.zeros_like(s_arr)

        K = [
            [one, s_arr, s_arr**2 / 2.0, s_arr**3 / 6.0],
            [zero, one, s_arr, s_arr**2 / 2.0],
            [zero, zero, one, s_arr],
            [zero, zero, zero, one],
        ]

        f = [zero, zero, zero, zero]

        return stack_transfer_eq(K, f)


@dataclass(slots=True)
class BeamElementStretched(TMMElement):
    """Stretched beam spline segment.

    Attributes
    ----------
    stretching_force
        A force that pulls from the edges of the beam.
        Higher values result in higher curvature but shorter length.
    """

    vectorized: ClassVar[bool] = True

    stretching_force: float

    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f."""
        s_arr = np.asarray(s, dtype=np.float64)
        one, zero = np.ones_like(s_arr), np.zeros_like(s_arr)

        N = self.stretching_force

        k = np.sqrt(N)

        ch_kx = np.cosh(k * s_arr)
        sh_kx = np.sinh(k * s_arr)

        K = [
            [one, s_arr, (ch_kx - 1) / N, (sh_kx - k * s_arr) / (k * N)],
            [zero, one, sh_kx / k, (ch_kx - 1) / N],
            [zero, zero, ch_kx, sh_kx / k],
            [zero, zero, k * sh_kx, ch_kx],
        ]

        f = [zero, zero, zero, zero]

        return stack_transfer_eq(K, f)


@dataclass(slots=True)
class BeamElementCompressed(TMMElement):
    """Compressed beam spline segment.

    Attributes
    ----------
    compressing_force
        A force that pushes from the edges of the beam.
        Higher values result in lower curvature but longer length.
    """

    vectorized: ClassVar[bool] = True

    compressing_force: float

    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f."""
        s_arr = np.asarray(s, dtype=np.float64)
        one, zero = np.ones_like(s_arr), np.zeros_like(s_arr)

        N = self.compressing_force

        k = np.sqrt(N)

        cos_kx = np.cos(k * s_arr)
        sin_kx = np.sin(k * s_arr)

        K = [
            [one, s_arr, (cos_kx - 1) / N, (sin_kx - k * s_arr) / (k * N)],
            [zero, one, sin_kx / k, (cos_kx - 1) / N],
            [zero, zero, cos_kx, sin_kx / k],
            [zero, zero, k * sin_kx, cos_kx],
        ]

        f = [zero, zero, zero, zero]

        return stack_transfer_eq(K, f)


@dataclass(slots=True)
class BeamElementOnElasticFoundation(TMMElement):
    """Smoothing beam spline segment on elastic foundation.

    Attributes
    ----------
    start_foundation
        Foundation displacement at the start of the beam.
    end_foundation
        Foundation displacement at the end of the beam.
    bandwidth
        A smoothing bandwidth (g) that controls the foundation stiffness.
    """

    vectorized: ClassVar[bool] = True

    start_foundation: float
    end_foundation: float
    bandwidth: float

    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f."""
        s_arr = np.asarray(s, dtype=np.float64)

        g = self.bandwidth

        xi = s_arr / np.sqrt(2) / g

        g2 = g * g
        g3 = g2 * g
        g4 = g3 * g

        cos_xi = np.cos(xi)
        sin_xi = np.sin(xi)

        ch_xi = np.cosh(xi)
        sh_xi = np.sinh(xi)

        k1 = ch_xi * cos_xi
        k2 = g / np.sqrt(2) * (sh_xi * cos_xi + ch_xi * sin_xi)
        k3 = g2 * sh_xi * sin_xi
        k4 = g3 / np.sqrt(2) * (ch_xi * sin_xi - sh_xi * cos_xi)

        K = [
            [k1, k2, k3, k4],
            [-k4 / g4, k1, k2, k3],
            [-k3 / g4, -k4 / g4, k1, k2],
            [-k2 / g4, -k3 / g4, -k4 / g4, k1],
        ]

        a = self.start_foundation
        b = (self.end_foundation - self.start_foundation) / self.length

        f = [
            a * (1 - k1) + b * (s_arr - k2),
            a / g4 * k4 + b * (1 - k1),
            a / g4 * k3 + b / g4 * k4,
            a / g4 * k2 + b / g4 * k3,
        ]

        return stack_transfer_eq(K, f)
