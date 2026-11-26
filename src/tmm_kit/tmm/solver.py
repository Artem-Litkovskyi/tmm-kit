from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import (
    ElementEdge,
    ElementGroups,
    TMMElement,
    TMMEquation,
    TMMNode,
    group_by_class,
    stack_fields,
)
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

        lengths = np.array([element.length for element in self.elements])
        tm_A, tm_b = ElementGroups(self.elements, self.state_size).get_transfer_eq(
            np.arange(elements_n), lengths
        )

        blocks[:, 0] = -tm_A
        f[:transfer_eqs_n] = tm_b.ravel()

        K_transfer = sp.bsr_matrix(
            (
                blocks.reshape(-1, self.state_size, self.state_size),
                np.arange(2 * elements_n),
                np.arange(0, 2 * elements_n + 1, 2),
            ),
            shape=(transfer_eqs_n, unknowns_n),
        )

        # Conjugation and boundary condition equations are added into the matrix
        # as (row, column, coefficient) triplets.
        # Repeated variables in the same equation are summed up.
        rows, cols, coefficients = [], [], []

        # The first equation row of each node, then of the boundary conditions
        node_rows = np.cumsum([0] + [n.get_conjugation_eq_n() for n in self.nodes])

        eqs, eq_rows = [], []
        for cls, node_indices in group_by_class(self.nodes).items():
            if not cls.vectorized:
                for node_i in node_indices:
                    node_eqs = self.nodes[node_i].get_conjugation_eq()
                    eqs.extend(node_eqs)
                    eq_rows.extend(range(node_rows[node_i], node_rows[node_i + 1]))
                continue

            # Each equation of the stacked node stands for one equation per node
            stacked = stack_fields([self.nodes[i] for i in node_indices])
            for eq_i, eq in enumerate(stacked.get_conjugation_eq()):
                group_rows = node_rows[node_indices] + eq_i
                f[transfer_eqs_n + group_rows] = eq.right_side

                for element_i, element_edge, state_var_i, k in eq.left_side_addends:
                    group_cols = self._global_matrix_index_of_var(
                        element_i, element_edge, state_var_i
                    )
                    rows.append(group_rows)
                    cols.append(np.broadcast_to(group_cols, group_rows.shape))
                    coefficients.append(np.broadcast_to(k, group_rows.shape))

        eqs.extend(self.bcs)
        eq_rows.extend(range(node_rows[-1], node_rows[-1] + len(self.bcs)))

        addend_rows, addends = [], []
        for row, eq in zip(eq_rows, eqs):
            addend_rows.extend([row] * len(eq.left_side_addends))
            addends.extend(eq.left_side_addends)
            f[transfer_eqs_n + row] = eq.right_side

        if addends:
            element_i, element_edge, state_var_i, k = zip(*addends)
            rows.append(addend_rows)
            cols.append(
                self._global_matrix_index_of_var(element_i, element_edge, state_var_i)
            )
            coefficients.append(k)

        K_eqs = sp.coo_matrix(
            (
                np.concatenate(coefficients, dtype=np.float64),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(unknowns_n - transfer_eqs_n, unknowns_n),
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
