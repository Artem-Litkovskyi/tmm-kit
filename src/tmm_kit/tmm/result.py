from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from .base import ElementEdge, ElementGroups, TMMElement


@dataclass
class TMMResult:
    """Immutable result container holding state vectors and evaluation logic."""

    elements: Sequence[TMMElement]
    edge_states: np.ndarray

    def __post_init__(self):
        """Calculate state size dynamically based on the flattened unknowns vector."""
        number_of_edges = 2 * len(self.elements)
        self.state_size = self.edge_states.size // number_of_edges

    def evaluate(
        self,
        element_i: npt.ArrayLike,
        s: npt.ArrayLike,
    ) -> np.ndarray:
        """Evaluate an element state at a certain local coordinate.

        Parameters
        ----------
        element_i
            Element index. Can be an array of indices that is broadcast
            with s: then each coordinate belongs to its own element.
        s
            Local coordinate (coordinates)
            from 0 to self.elements[element_i].length.

        Returns
        -------
        states
            States of shape ``(*s.shape, state_size)``
            (the broadcast shape if element_i is an array).
        """
        s_arr = np.asarray(s, dtype=np.float64)

        if np.ndim(element_i) > 0:
            return self._evaluate_elements(np.asarray(element_i, dtype=np.int64), s_arr)

        tm_A, tm_b = self.elements[element_i].get_transfer_eq(s_arr)
        states = tm_A @ self.get_edge_state(element_i, ElementEdge.START) + tm_b

        expected_shape = (*s_arr.shape, self.state_size)
        if states.shape != expected_shape:
            raise ValueError(
                f"get_transfer_eq of element {element_i} returned states "
                f"of shape {states.shape} instead of {expected_shape}."
            )
        return states

    def get_edge_state(
        self,
        element_i: npt.ArrayLike,
        element_edge: ElementEdge | None = None,
    ) -> np.ndarray:
        """Get state vectors from the array of all element states.

        Parameters
        ----------
        element_i
            Element index (indices).
        element_edge
            The edge at which the state is retrieved.
            If None, retrieve at both edges.
        """
        element_indices = np.asarray(element_i, dtype=np.int64)
        states = self._states_by_element()

        if element_edge is None:
            return states[element_indices].reshape(-1, self.state_size)
        return states[element_indices, self._edge_index_offset(element_edge)]

    def get_all_edge_states(
        self,
        element_edge: ElementEdge | None = None,
    ) -> np.ndarray:
        """Get all state vectors from the array of all element states.

        Parameters
        ----------
        element_edge
            The edge at which the state is retrieved.
            If None, retrieve at both edges.
        """
        if element_edge is None:
            return self.edge_states.reshape(-1, self.state_size)
        return self._states_by_element()[:, self._edge_index_offset(element_edge)]

    def _evaluate_elements(self, element_i: np.ndarray, s: np.ndarray) -> np.ndarray:
        element_i, s = np.broadcast_arrays(element_i, s)
        flat_element_i = element_i.ravel()

        tm_A, tm_b = self._element_groups.get_transfer_eq(flat_element_i, s.ravel())
        start_states = self.get_edge_state(flat_element_i, ElementEdge.START)
        states = np.einsum("nij,nj->ni", tm_A, start_states) + tm_b

        return states.reshape(*s.shape, self.state_size)

    @cached_property
    def _element_groups(self) -> ElementGroups:
        return ElementGroups(self.elements, self.state_size)

    def _states_by_element(self) -> np.ndarray:
        return self.edge_states.reshape(len(self.elements), 2, self.state_size)

    @staticmethod
    def _edge_index_offset(element_edge: ElementEdge) -> int:
        return int(element_edge == ElementEdge.END)
