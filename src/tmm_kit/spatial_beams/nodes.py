from collections.abc import Sequence
from dataclasses import dataclass

from tmm_kit.spatial_beams.constants import (
    SPATIAL_BEAM_DISPLACEMENT,
    SPATIAL_BEAM_FORCE,
    SPATIAL_BEAM_MOMENT,
    SPATIAL_BEAM_ROTATION,
    SPATIAL_BEAM_STATE_SIZE,
)
from tmm_kit.spatial_beams.elements import SpatialBeamElement
from tmm_kit.tmm.base import ElementEdge, TMMEquation, TMMEquationAddend, TMMNode

_AXES = range(3)


@dataclass(slots=True)
class SpatialBeamRigidNode(TMMNode):
    """Rigid joint of any number of spatial beam elements.

    Enforce continuity of the displacement and rotation vectors
    and equilibrium of the force and moment vectors in global coordinates.

    Attributes
    ----------
    elements
        All elements of the system (the same sequence that is passed
        to the solver). Used to read the bases of the connected elements.
    """

    elements: Sequence[SpatialBeamElement]

    def get_conjugation_eq(self) -> list[TMMEquation]:
        """Return the conjugation equations connecting elements."""
        # (element_i, edge, sign of the force flowing out of the node)
        edges = [
            *((el_i, ElementEdge.END, -1.0) for el_i in self.inbound_elements),
            *((el_i, ElementEdge.START, 1.0) for el_i in self.outbound_elements),
        ]
        ref_el, ref_edge, _ = edges[0]

        eqs = []

        # Continuity with the reference edge
        for group in (SPATIAL_BEAM_DISPLACEMENT, SPATIAL_BEAM_ROTATION):
            for el_i, edge, _ in edges[1:]:
                for global_axis in _AXES:
                    eqs.append(
                        TMMEquation(
                            [
                                *self._global_component(
                                    el_i, edge, group, global_axis, 1.0
                                ),
                                *self._global_component(
                                    ref_el, ref_edge, group, global_axis, -1.0
                                ),
                            ],
                            0,
                        )
                    )

        # Equilibrium: outbound minus inbound
        for group in (SPATIAL_BEAM_MOMENT, SPATIAL_BEAM_FORCE):
            for global_axis in _AXES:
                addends = []
                for el_i, edge, sign in edges:
                    addends.extend(
                        self._global_component(el_i, edge, group, global_axis, sign)
                    )
                eqs.append(TMMEquation(addends, 0))

        return eqs

    def get_conjugation_eq_n(self) -> int:
        """Return the number of conjugation equations."""
        edges_n = len(self.inbound_elements) + len(self.outbound_elements)
        return SPATIAL_BEAM_STATE_SIZE // 2 * edges_n

    def _global_component(
        self,
        element_i: int,
        edge: ElementEdge,
        group: int,
        global_axis: int,
        sign: float,
    ) -> list[TMMEquationAddend]:
        """Return addends of a global component of a local vector."""
        basis = self.elements[element_i].basis
        return [
            (element_i, edge, group + local_axis, sign * basis[local_axis, global_axis])
            for local_axis in _AXES
        ]
