from dataclasses import dataclass

import numpy as np

from .base import ElementEdge, TMMElement


@dataclass
class TMMResult:
    """Immutable result container holding state vectors and evaluation logic."""

    elements: list[TMMElement]
    edge_states: np.ndarray

    def __post_init__(self):
        """Calculate state size dynamically based on the flattened unknowns vector."""
        number_of_edges = 2 * len(self.elements)
        self.state_size = self.edge_states.size // number_of_edges

    def evaluate(
        self,
        element_i: int,
        s: np.typing.ArrayLike,
    ) -> np.ndarray:
        """Evaluate an element state at a certain local coordinate.

        Parameters
        ----------
        element_i
            Element index.
        s
            Local coordinate (coordinates)
            from 0 to self.elements[element_i].length.
        """
        eval_s_arr = np.atleast_1d(np.asarray(s, dtype=np.float64))
        states = np.empty((eval_s_arr.size, self.state_size), dtype=np.float64)

        for i, eval_s in enumerate(eval_s_arr):
            start_state = self.get_edge_state(element_i, ElementEdge.START)
            tm_A, tm_b = self.elements[element_i].get_transfer_eq(eval_s)
            states[i] = tm_A @ start_state + tm_b

        if np.isscalar(s):
            return states[0]
        return states

    def get_edge_state(
        self,
        element_i: np.typing.ArrayLike,
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
        element_indices = np.atleast_1d(np.asarray(element_i, dtype=np.int64))
        states = self.edge_states.reshape(-1, self.state_size)

        if element_edge is None:
            state_indices = np.stack(
                (2 * element_indices, 2 * element_indices + 1),
                axis=1,
            ).reshape(-1)
            return states[state_indices]

        edge_offset = int(element_edge == ElementEdge.END)
        state_indices = 2 * element_indices + edge_offset
        selected_states = states[state_indices]

        if np.isscalar(element_i):
            return selected_states[0]
        return selected_states

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
        return self.get_edge_state(range(len(self.elements)), element_edge)
