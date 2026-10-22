import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .base import ElementEdge, TMMElement, TMMEquation, TMMNode
from .result import TMMResult


class TMMSolver:
    """Transfer Matrix Method Solver."""

    def __init__(
        self,
        state_size: int,
        elements: list[TMMElement],
        nodes: list[TMMNode],
        boundary_conditions: list[TMMEquation],
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

        K = sp.lil_matrix((unknowns_n, unknowns_n), dtype=np.float64)
        f = np.zeros(unknowns_n, dtype=np.float64)

        row = 0

        # Transfer equations
        for el_i, element in enumerate(self.elements):
            tm_A, tm_b = element.get_transfer_eq(element.length)

            idx_l = self._global_matrix_index_of_state(el_i, ElementEdge.START)
            idx_r = self._global_matrix_index_of_state(el_i, ElementEdge.END)

            K[row : row + self.state_size, idx_l : idx_l + self.state_size] = -tm_A
            K[row : row + self.state_size, idx_r : idx_r + self.state_size] = np.eye(
                self.state_size
            )

            f[row : row + self.state_size] = tm_b

            row += self.state_size

        eqs = []

        # Conjugation equations
        for node in self.nodes:
            eqs.extend(node.get_conjugation_eq())

        # Boundary condition equations
        eqs.extend(self.bcs)

        # Add equations into the matrix
        for eq in eqs:
            for seg_i, seg_side, state_var, k in eq.left_side_addends:
                idx = self._global_matrix_index_of_var(seg_i, seg_side, state_var)
                K[row, idx] = k
            f[row] = eq.right_side
            row += 1

        return sp.csr_matrix(K), f

    def _calculate_unknowns_n(self):
        return 2 * self.state_size * len(self.elements)

    def _calculate_equations_n(self):
        transfer_eqs_n = self.state_size * len(self.elements)
        conjugation_eqs_n = sum([n.get_conjugation_eq_n() for n in self.nodes])
        bcs_n = len(self.bcs)
        return transfer_eqs_n, conjugation_eqs_n, bcs_n

    def _global_matrix_index_of_state(
        self, element_i: int, element_edge: ElementEdge
    ) -> int:
        """Calculate index in the global matrix for a beginning of a specific state."""
        edge_offset = int(element_edge == ElementEdge.END)
        return self.state_size * (2 * element_i + edge_offset)

    def _global_matrix_index_of_var(
        self, element_i: int, element_edge: ElementEdge, state_var_i: int
    ) -> int:
        """Calculate index in the global matrix for a specific state variable."""
        return self._global_matrix_index_of_state(element_i, element_edge) + state_var_i
