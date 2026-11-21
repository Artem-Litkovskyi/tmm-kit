from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tmm_kit.tmm.base import TMMElement, stack_transfer_eq


@dataclass(slots=True)
class ElectricalElement(TMMElement):
    """Electrical circuit element."""

    resistance: float
    voltage: float

    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f."""
        s_arr = np.asarray(s, dtype=np.float64)
        one, zero = np.ones_like(s_arr), np.zeros_like(s_arr)

        K = [
            [one, -self.resistance * one],
            [zero, one],
        ]

        f = [
            self.voltage * one,
            zero,
        ]

        return stack_transfer_eq(K, f)
