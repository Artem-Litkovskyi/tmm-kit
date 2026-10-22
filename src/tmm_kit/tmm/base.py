from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np


class ElementEdge(StrEnum):
    """Edge of an 1D element."""

    START = "start"
    END = "end"


# Format: (element_i, ElementEdge, state_var_i, coefficient)
TMMEquationAddend: TypeAlias = tuple[int, ElementEdge, int, float]


@dataclass(init=False)
class TMMEquation:
    """An equation that can be added to the TMM system of equations."""

    def __init__(
        self,
        left_side: TMMEquationAddend | Sequence[TMMEquationAddend],
        right_side: float,
    ):
        if isinstance(left_side, Sequence) and isinstance(left_side[0], tuple):
            self.left_side_addends = left_side
        else:
            self.left_side_addends = [left_side]
        self.right_side = right_side


@dataclass(slots=True)
class TMMElement(ABC):
    """Abstract base class for all transfer matrix elements."""

    length: float

    @abstractmethod
    def get_transfer_eq(self, s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f.

        Parameters
        ----------
        s
            Local coordinate from 0 to self.length.
        """
        pass


@dataclass(slots=True)
class TMMNode(ABC):
    """Abstract base class for nodes conjugating elements."""

    inbound_elements: list[int]
    outbound_elements: list[int]

    @abstractmethod
    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        pass

    @abstractmethod
    def get_conjugation_eq_n(self) -> int:
        """Return the number of conjugation equations."""
        pass


def offset_equation(in_el: int, out_el: int, var_i: int, offset: float):
    """Return an equation with an offset between adjacent element edges."""
    return TMMEquation(
        [
            (out_el, ElementEdge.START, var_i, 1),
            (in_el, ElementEdge.END, var_i, -1),
        ],
        offset,
    )


def continuity_equation(in_el: int, out_el: int, var_i: int):
    """Return continuity equation."""
    return offset_equation(in_el, out_el, var_i, 0.0)
