from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import ClassVar, TypeAlias

import numpy as np
import numpy.typing as npt


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
    """Abstract base class for all transfer matrix elements.

    Notes
    -----
    Set the class variable ``vectorized = True`` to let the solver process
    all elements of the class in a single `get_transfer_eq` call. The call
    is made on an element whose fields are arrays of shape ``(N,)`` with
    the values of N elements, and s has the same shape. Thus the method
    must use only element-wise operations on the fields.
    """

    vectorized: ClassVar[bool] = False

    length: float

    @abstractmethod
    def get_transfer_eq(self, s: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the transfer matrix K and right vector f.

        The state at the local coordinate s is ``K @ start_state + f``.

        Parameters
        ----------
        s
            Local coordinate (coordinates) from 0 to self.length.

        Returns
        -------
        K
            Transfer matrices of shape ``(*s.shape, S, S)``,
            where S is the state size. A scalar s gives ``(S, S)``.
        f
            Right vectors of shape ``(*s.shape, S)``.
            A scalar s gives ``(S,)``.

        Notes
        -----
        Use `stack_transfer_eq` to build K and f from entries
        that are arrays of the same shape as s.
        """
        pass


def stack_transfer_eq(
    K: Sequence[Sequence[npt.ArrayLike]],
    f: Sequence[npt.ArrayLike],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack transfer equation entries into arrays of matrices and vectors.

    Parameters
    ----------
    K
        Nested ``S x S`` matrix whose entries are arrays of the same shape as s.
    f
        Vector of S entries that are arrays of the same shape as s.

    Returns
    -------
    K
        Transfer matrices of shape ``(*s.shape, S, S)``.
    f
        Right vectors of shape ``(*s.shape, S)``.
    """
    K_arr = np.asarray(K, dtype=np.float64)
    f_arr = np.asarray(f, dtype=np.float64)

    # Move the entry axes to the end. Cheaper than np.moveaxis for scalar s,
    # which matters because the solver calls this once per element.
    return (
        K_arr.transpose(*range(2, K_arr.ndim), 0, 1),
        f_arr.transpose(*range(1, f_arr.ndim), 0),
    )


@dataclass(slots=True)
class TMMNode(ABC):
    """Abstract base class for nodes conjugating elements.

    Notes
    -----
    Set the class variable ``vectorized = True`` to let the solver process
    all nodes of the class in a single `get_conjugation_eq` call. The call
    is made on a node whose fields are arrays with the values of N nodes
    stacked along the last axis (e.g., ``inbound_elements[0]`` has shape
    ``(N,)``). The element indices, coefficients and right sides of the returned
    equations can be such arrays: each equation is added once per node.
    All nodes of the class must return the same number of equations.
    """

    vectorized: ClassVar[bool] = False

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


def stack_fields[T](objects: Sequence[T]) -> T:
    """Stack dataclass instances of the same class into a single instance.

    Each field of the returned instance is an array of the field values
    of all the instances, stacked along the last axis.
    """
    stacked = copy(objects[0])
    for field in fields(stacked):
        values = np.asarray([getattr(obj, field.name) for obj in objects])
        setattr(stacked, field.name, np.moveaxis(values, 0, -1))
    return stacked


def group_by_class(objects: Sequence) -> dict[type, list[int]]:
    """Return indices of the objects for each of their classes."""
    groups = {}
    for i, obj in enumerate(objects):
        groups.setdefault(type(obj), []).append(i)
    return groups


def take_fields[T](stacked: T, indices: np.ndarray) -> T:
    """Take the instances with the given indices from a stacked instance."""
    taken = copy(stacked)
    for field in fields(stacked):
        setattr(taken, field.name, getattr(stacked, field.name)[..., indices])
    return taken


class ElementGroups:
    """Elements grouped by class for evaluating many transfer equations at once.

    Elements of a vectorized class are processed in a single call.

    Parameters
    ----------
    elements
        All elements of the system.
    state_size
        State size S.
    """

    def __init__(self, elements: Sequence[TMMElement], state_size: int):
        self.elements = elements
        self.state_size = state_size

        # Group index of each element and its position in the group
        self._group_i = np.empty(len(elements), dtype=np.int64)
        self._position = np.empty(len(elements), dtype=np.int64)
        # Stacked elements of each group (None if the class is not vectorized)
        self._stacked = []

        for group_i, (cls, indices) in enumerate(group_by_class(elements).items()):
            self._group_i[indices] = group_i
            self._position[indices] = np.arange(len(indices))
            self._stacked.append(
                stack_fields([elements[i] for i in indices]) if cls.vectorized else None
            )

    def get_transfer_eq(
        self, element_i: np.ndarray, s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return transfer equations of elements, each at its own coordinate.

        Parameters
        ----------
        element_i
            Element indices of shape ``(N,)``. An index can be repeated.
        s
            Local coordinates of shape ``(N,)``, one per element index.

        Returns
        -------
        K
            Transfer matrices of shape ``(N, S, S)``.
        f
            Right vectors of shape ``(N, S)``.
        """
        K = np.empty((s.size, self.state_size, self.state_size), dtype=np.float64)
        f = np.empty((s.size, self.state_size), dtype=np.float64)

        group_i = self._group_i[element_i]
        for i, stacked in enumerate(self._stacked):
            group = np.flatnonzero(group_i == i)

            if stacked is not None:
                taken = take_fields(stacked, self._position[element_i[group]])
                K[group], f[group] = taken.get_transfer_eq(s[group])
                continue

            # A repeated element is called once for all its coordinates
            for el_i in np.unique(element_i[group]):
                of_element = group[element_i[group] == el_i]
                tm_K, tm_f = self.elements[el_i].get_transfer_eq(s[of_element])

                expected_shape = (of_element.size, self.state_size)
                if tm_f.shape != expected_shape:
                    raise ValueError(
                        f"get_transfer_eq of element {el_i} returned right vectors "
                        f"of shape {tm_f.shape} instead of {expected_shape}."
                    )
                K[of_element], f[of_element] = tm_K, tm_f

        return K, f
