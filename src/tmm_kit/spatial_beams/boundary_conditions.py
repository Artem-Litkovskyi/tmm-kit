from tmm_kit.spatial_beams.constants import (
    SPATIAL_BEAM_DISPLACEMENT,
    SPATIAL_BEAM_FORCE,
    SPATIAL_BEAM_MOMENT,
    SPATIAL_BEAM_ROTATION,
)
from tmm_kit.tmm.base import ElementEdge, TMMEquation

_AXES = range(3)


def pinned_end(element_i: int, edge: ElementEdge) -> list[TMMEquation]:
    """Pinned support at an element edge (zero displacements and moments)."""
    return [
        *_zero_vector(element_i, edge, SPATIAL_BEAM_DISPLACEMENT),
        *_zero_vector(element_i, edge, SPATIAL_BEAM_MOMENT),
    ]


def clamped_end(element_i: int, edge: ElementEdge) -> list[TMMEquation]:
    """Clamped support at an element edge (zero displacements and rotations)."""
    return [
        *_zero_vector(element_i, edge, SPATIAL_BEAM_DISPLACEMENT),
        *_zero_vector(element_i, edge, SPATIAL_BEAM_ROTATION),
    ]


def free_end(element_i: int, edge: ElementEdge) -> list[TMMEquation]:
    """No support at an element edge (zero forces and moments)."""
    return [
        *_zero_vector(element_i, edge, SPATIAL_BEAM_FORCE),
        *_zero_vector(element_i, edge, SPATIAL_BEAM_MOMENT),
    ]


def sliding_end(element_i: int, edge: ElementEdge, axis: int) -> list[TMMEquation]:
    """Sliding support at an element edge.

    Constrains the displacement along one local axis. The force components
    along the two other local axes and all moments are zero.

    Parameters
    ----------
    element_i
        Element index.
    edge
        Supported edge of the element.
    axis
        Local axis of the constrained displacement
        (``AXIS_T``, ``AXIS_N`` or ``AXIS_B``).
    """
    return [
        TMMEquation((element_i, edge, SPATIAL_BEAM_DISPLACEMENT + axis, 1), 0),
        *(
            TMMEquation((element_i, edge, SPATIAL_BEAM_FORCE + free_axis, 1), 0)
            for free_axis in _AXES
            if free_axis != axis
        ),
        *_zero_vector(element_i, edge, SPATIAL_BEAM_MOMENT),
    ]


def _zero_vector(element_i: int, edge: ElementEdge, group: int) -> list[TMMEquation]:
    return [TMMEquation((element_i, edge, group + axis, 1), 0) for axis in _AXES]
