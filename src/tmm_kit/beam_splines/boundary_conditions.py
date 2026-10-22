from tmm_kit.beam_splines.constants import (
    BEAM_DISPLACEMENT,
    BEAM_FORCE,
    BEAM_MOMENT,
    BEAM_SLOPE,
)
from tmm_kit.tmm.base import (
    ElementEdge,
    TMMEquation,
    continuity_equation,
    offset_equation,
)


def pinned_ends(start_displacement, end_displacement):
    """Pinned supports at both ends (zero moments)."""
    return [
        TMMEquation((0, ElementEdge.START, BEAM_DISPLACEMENT, 1), start_displacement),
        TMMEquation((-1, ElementEdge.END, BEAM_DISPLACEMENT, 1), end_displacement),
        TMMEquation((0, ElementEdge.START, BEAM_MOMENT, 1), 0),
        TMMEquation((-1, ElementEdge.END, BEAM_MOMENT, 1), 0),
    ]


def clamped_ends(
    start_displacement,
    end_displacement,
    start_slope,
    end_slope,
):
    """Clamped supports at both ends."""
    return [
        TMMEquation((0, ElementEdge.START, BEAM_DISPLACEMENT, 1), start_displacement),
        TMMEquation((-1, ElementEdge.END, BEAM_DISPLACEMENT, 1), end_displacement),
        TMMEquation((0, ElementEdge.START, BEAM_SLOPE, 1), start_slope),
        TMMEquation((-1, ElementEdge.END, BEAM_SLOPE, 1), end_slope),
    ]


def elastic_ends(
    start_displacement,
    end_displacement,
    start_stiffness,
    end_stiffness,
):
    """Elastic supports at both ends (zero moments, spring forces)."""
    return [
        TMMEquation((0, ElementEdge.START, BEAM_MOMENT, 1), 0),
        TMMEquation((-1, ElementEdge.END, BEAM_MOMENT, 1), 0),
        TMMEquation(
            [
                (0, ElementEdge.START, BEAM_FORCE, 1),
                (0, ElementEdge.START, BEAM_DISPLACEMENT, start_stiffness),
            ],
            start_stiffness * start_displacement,
        ),
        TMMEquation(
            [
                (-1, ElementEdge.END, BEAM_FORCE, 1),
                (-1, ElementEdge.END, BEAM_DISPLACEMENT, -end_stiffness),
            ],
            -end_stiffness * end_displacement,
        ),
    ]


def elastic_ends_inverted(
    start_displacement,
    end_displacement,
    start_compliance,
    end_compliance,
):
    """Inverted elastic supports at both ends (zero moments, spring forces)."""
    return [
        TMMEquation((0, ElementEdge.START, BEAM_MOMENT, 1), 0),
        TMMEquation((-1, ElementEdge.END, BEAM_MOMENT, 1), 0),
        TMMEquation(
            [
                (0, ElementEdge.START, BEAM_FORCE, start_compliance),
                (0, ElementEdge.START, BEAM_DISPLACEMENT, 1),
            ],
            start_displacement,
        ),
        TMMEquation(
            [
                (-1, ElementEdge.END, BEAM_FORCE, end_compliance),
                (-1, ElementEdge.END, BEAM_DISPLACEMENT, -1),
            ],
            -end_displacement,
        ),
    ]


def free_ends():
    """No supports at both ends (zero moments and forces)."""
    return [
        TMMEquation((0, ElementEdge.START, BEAM_MOMENT, 1), 0),
        TMMEquation((-1, ElementEdge.END, BEAM_MOMENT, 1), 0),
        TMMEquation((0, ElementEdge.START, BEAM_FORCE, 1), 0),
        TMMEquation((-1, ElementEdge.END, BEAM_FORCE, 1), 0),
    ]


def pinned_loop(displacement, angle_misalignment=0):
    """Pinned support for a looped spline."""
    return [
        TMMEquation((0, ElementEdge.START, BEAM_DISPLACEMENT, 1), displacement),
        TMMEquation((-1, ElementEdge.END, BEAM_DISPLACEMENT, 1), displacement),
        offset_equation(-1, 0, BEAM_SLOPE, angle_misalignment),
        continuity_equation(-1, 0, BEAM_MOMENT),
    ]


def clamped_loop(displacement, slope, angle_misalignment=0):
    """Clamped support for a looped spline."""
    return [
        TMMEquation((0, ElementEdge.START, BEAM_DISPLACEMENT, 1), displacement),
        TMMEquation((-1, ElementEdge.END, BEAM_DISPLACEMENT, 1), displacement),
        TMMEquation((0, ElementEdge.START, BEAM_SLOPE, 1), slope),
        offset_equation(-1, 0, BEAM_SLOPE, angle_misalignment),
    ]


def elastic_loop(displacement, stiffness, angle_misalignment=0):
    """Elastic support for a looped spline."""
    return [
        continuity_equation(-1, 0, BEAM_DISPLACEMENT),
        offset_equation(-1, 0, BEAM_SLOPE, angle_misalignment),
        continuity_equation(-1, 0, BEAM_MOMENT),
        TMMEquation(
            [
                (0, ElementEdge.START, BEAM_FORCE, 1),
                (-1, ElementEdge.END, BEAM_FORCE, -1),
                (-1, ElementEdge.END, BEAM_DISPLACEMENT, stiffness),
            ],
            stiffness * displacement,
        ),
    ]


def elastic_loop_inverted(displacement, compliance, angle_misalignment=0):
    """Inverted elastic support for a looped spline."""
    return [
        continuity_equation(-1, 0, BEAM_DISPLACEMENT),
        offset_equation(-1, 0, BEAM_SLOPE, angle_misalignment),
        continuity_equation(-1, 0, BEAM_MOMENT),
        TMMEquation(
            [
                (0, ElementEdge.START, BEAM_FORCE, 1),
                (-1, ElementEdge.END, BEAM_FORCE, -compliance),
                (-1, ElementEdge.END, BEAM_DISPLACEMENT, compliance),
            ],
            displacement,
        ),
    ]
