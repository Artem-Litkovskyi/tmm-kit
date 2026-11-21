from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import ElementEdge, TMMElement, TMMEquation, TMMNode
from .result import TMMResult


class TMMSolver:
    """Transfer Matrix Method Solver."""

    def __init__(
        self,
        state_size: int,
        elements: Sequence[TMMElement],
        nodes: Sequence[TMMNode],
        boundary_conditions: Sequence[TMMEquation],
    ):
        if state_size < 1:
            raise ValueError("state_size must be at least 1.")

        if not elements:
            raise ValueError("At least one TMMElement must be provided.")

        self.state_size = state_size
        self.elements = elements
        self.nodes = nodes
        self.bcs = boundary_conditions

        unknowns_n = self._calculate_unknowns_n()
        transfer_eqs_n, conjugation_eqs_n, bcs_n = self._calculate_equations_n()
        equations_n = transfer_eqs_n + conjugation_eqs_n + bcs_n
        if unknowns_n != equations_n:
            raise ValueError(
                f"The number of unknowns ({unknowns_n}) "
                f"is not equal to the total number of equations ({equations_n}): "
                f"{transfer_eqs_n} transfer equations, "
                f"{conjugation_eqs_n} conjugation equations, "
                f"{bcs_n} boundary conditions."
            )

    def solve(self) -> TMMResult:
        """Solve the boundary value problem and return a TMMResult."""
        global_matrix, right_vector = self._assemble_system()

        edge_states = spla.spsolve(global_matrix, right_vector)

        return TMMResult(
            elements=self.elements,
            edge_states=edge_states,
        )

    def _assemble_system(self) -> tuple[sp.csr_matrix, np.ndarray]:
        unknowns_n = self._calculate_unknowns_n()
        elements_n = len(self.elements)
        transfer_eqs_n = self.state_size * elements_n

        f = np.zeros(unknowns_n, dtype=np.float64)

        # Transfer equations
        # Block row i holds [-A_i, I] in the block columns (2i, 2i + 1)
        blocks = np.empty(
            (elements_n, 2, self.state_size, self.state_size), dtype=np.float64
        )
        blocks[:, 1] = np.eye(self.state_size)

        for el_i, element in enumerate(self.elements):
            tm_A, tm_b = element.get_transfer_eq(element.length)

            blocks[el_i, 0] = -tm_A

            row = self.state_size * el_i
            f[row : row + self.state_size] = tm_b

        K_transfer = sp.bsr_matrix(
            (
                blocks.reshape(-1, self.state_size, self.state_size),
                np.arange(2 * elements_n),
                np.arange(0, 2 * elements_n + 1, 2),
            ),
            shape=(transfer_eqs_n, unknowns_n),
        )

        eqs = []

        # Conjugation equations
        for node in self.nodes:
            eqs.extend(node.get_conjugation_eq())

        # Boundary condition equations
        eqs.extend(self.bcs)

        # Add equations into the matrix as (row, column, coefficient) triplets.
        # Repeated variables in the same equation are summed up.
        rows, addends = [], []
        for row, eq in enumerate(eqs):
            rows.extend([row] * len(eq.left_side_addends))
            addends.extend(eq.left_side_addends)
            f[transfer_eqs_n + row] = eq.right_side

        element_i, element_edge, state_var_i, k = zip(*addends)
        cols = self._global_matrix_index_of_var(element_i, element_edge, state_var_i)

        K_eqs = sp.coo_matrix(
            (k, (rows, cols)), shape=(len(eqs), unknowns_n), dtype=np.float64
        )

        K = sp.vstack((K_transfer, K_eqs), format="csr")
        K.eliminate_zeros()

        return K, f

    def _calculate_unknowns_n(self):
        return 2 * self.state_size * len(self.elements)

    def _calculate_equations_n(self):
        transfer_eqs_n = self.state_size * len(self.elements)
        conjugation_eqs_n = sum([n.get_conjugation_eq_n() for n in self.nodes])
        bcs_n = len(self.bcs)
        return transfer_eqs_n, conjugation_eqs_n, bcs_n

    def _global_matrix_index_of_var(
        self,
        element_i: npt.ArrayLike,
        element_edge: npt.ArrayLike,
        state_var_i: npt.ArrayLike,
    ) -> np.ndarray:
        """Calculate global matrix indices of state variables.

        Negative element indices count from the end, as in Python.
        """
        element_i_arr = np.asarray(element_i, dtype=np.int64)
        element_i_arr = np.where(
            element_i_arr < 0, element_i_arr + len(self.elements), element_i_arr
        )
        edge_offset = np.asarray(element_edge) == ElementEdge.END
        state_var_i_arr = np.asarray(state_var_i, dtype=np.int64)
        return self.state_size * (2 * element_i_arr + edge_offset) + state_var_i_arr
