from collections.abc import Sequence

import numpy as np

from tmm_kit import TMMSolver
from tmm_kit.beam_splines import (
    BEAM_STATE_SIZE,
    BeamElement,
    BeamPinnedSupport,
    CorotationalBeamSpline,
    angle_misalignment,
    create_cbs_elements_and_nodes,
    pinned_ends,
    pinned_loop,
)

# Polylines
SIMPLE_POLYLINE = [[0, 0], [1, 1], [2, 1], [3, 2]]
CIRCLE_POLYLINE = [[-2, 1], [2, 1], [2, -1], [-2, -1], [-2, 1]]

# Evaluation points
CIRCLE_EVAL_T = np.linspace(0, 12, 256)

# Desired results
DESIRED_SIMPLE_ANGLE_DIFFS = [0, np.pi / 4, -np.pi / 4, 0]
DESIRED_SIMPLE_POLYLINE_T = [0, np.sqrt(2), 1 + np.sqrt(2), 1 + 2 * np.sqrt(2)]
DESIRED_RADIUS = np.sqrt(5)
DESIRED_CURVATURE = 1 / DESIRED_RADIUS


def set_up_solver(
    polyline,
    loop=False,
    dummy_nodes_per_element: int | Sequence[int] = 0,
):
    elements, nodes, support_t, angle_diffs = create_cbs_elements_and_nodes(
        polyline=polyline,
        element_class=BeamElement,
        node_class=BeamPinnedSupport,
        dummy_nodes_per_element=dummy_nodes_per_element,
    )

    if loop:
        bcs = pinned_loop(0, angle_diffs[0])
    else:
        bcs = pinned_ends(0, 0)

    solver = TMMSolver(
        BEAM_STATE_SIZE,
        elements,
        nodes,
        bcs,
    )

    return solver, support_t


def test_angle_misalignment():
    angle_diffs = angle_misalignment(SIMPLE_POLYLINE)

    np.testing.assert_allclose(
        angle_diffs,
        DESIRED_SIMPLE_ANGLE_DIFFS,
    )


def test_factory_support_t():
    _, support_t = set_up_solver(SIMPLE_POLYLINE)

    np.testing.assert_allclose(
        support_t,
        DESIRED_SIMPLE_POLYLINE_T,
    )


def test_evaluate_at_polyline():
    solver, support_t = set_up_solver(SIMPLE_POLYLINE)
    spline = CorotationalBeamSpline(solver.solve(), support_t, SIMPLE_POLYLINE)

    points = spline.evaluate_spline_point(DESIRED_SIMPLE_POLYLINE_T)

    np.testing.assert_allclose(
        points,
        SIMPLE_POLYLINE,
    )


def test_approximate_circle_no_dummy_nodes():
    solver, support_t = set_up_solver(CIRCLE_POLYLINE, loop=True)
    spline = CorotationalBeamSpline(solver.solve(), support_t, CIRCLE_POLYLINE)

    points = spline.evaluate_spline_point(CIRCLE_EVAL_T)
    radius = np.linalg.norm(points, axis=1)

    np.testing.assert_allclose(
        radius,
        DESIRED_RADIUS,
        rtol=1e-1,
    )


def test_approximate_circle_one_dummy_node_improves_accuracy():
    solver, support_t = set_up_solver(CIRCLE_POLYLINE, loop=True)
    spline = CorotationalBeamSpline(solver.solve(), support_t, CIRCLE_POLYLINE)

    points = spline.evaluate_spline_point(CIRCLE_EVAL_T)
    radius = np.linalg.norm(points, axis=1)
    radius_error1 = np.max(abs(radius - DESIRED_RADIUS))

    dummy_nodes_per_element = 1
    new_polyline = spline.polyline_for_dummies(dummy_nodes_per_element)

    solver, support_t = set_up_solver(new_polyline, True, dummy_nodes_per_element)
    spline = CorotationalBeamSpline(solver.solve(), support_t, new_polyline)

    eval_t = np.linspace(support_t[0], support_t[-1], 256)

    points = spline.evaluate_spline_point(eval_t)
    radius = np.linalg.norm(points, axis=1)
    radius_error2 = np.max(abs(radius - DESIRED_RADIUS))

    np.testing.assert_array_less(
        radius_error2,
        radius_error1,
    )
