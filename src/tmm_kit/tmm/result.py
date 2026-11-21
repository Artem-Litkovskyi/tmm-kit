from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .base import ElementEdge, TMMElement


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
        element_i: int,
        s: npt.ArrayLike,
    ) -> np.ndarray:
        """Evaluate an element state at a certain local coordinate.

        Parameters
        ----------
        element_i
            Element index.
        s
            Local coordinate (coordinates)
            from 0 to self.elements[element_i].length.

        Returns
        -------
        states
            States of shape ``(*s.shape, state_size)``.
        """
        s_arr = np.asarray(s, dtype=np.float64)

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

    def _states_by_element(self) -> np.ndarray:
        return self.edge_states.reshape(len(self.elements), 2, self.state_size)

    @staticmethod
    def _edge_index_offset(element_edge: ElementEdge) -> int:
        return int(element_edge == ElementEdge.END)
