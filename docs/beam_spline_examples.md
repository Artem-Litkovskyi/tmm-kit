from tmm_kit.beam_splines import centered_element_length

# Beam Spline Code Example

This document gives a couple of examples of how to use `TMMSolver`, `BeamSpline` and `CorotationalBeamSpline`.

It doesn't cover all abilities of this library (e.g., different element, node and boundary condition presets).
You can read the docstrings in the source code for more details. 

## Simple Beam Spline

### Interpolation

Elements, nodes and boundary conditions are instantiated either by factory functions:

```python
import tmm_kit.beam_splines as bs

# Three supports (two elements conjugated at one node)
support_t = [2, 4, 5]  # Support position
support_w = [-1, 4, 2]  # Support displacement

# Use elements and nodes factory
elements, nodes = bs.create_beam_elements_and_nodes(
    support_t=support_t,
    element_class=bs.BeamElement,
    node_class=bs.BeamPinnedSupport,
    node_data={"displacement": support_w[1:-1]},
)

# Use boundary conditions factory
bcs = bs.pinned_ends(support_w[0], support_w[-1])
```

or manually, which is verbose, but gives more flexibility:

<details>

```python
import tmm_kit.beam_splines as bs
from tmm_kit import ElementEdge, TMMEquation

elements = [
    bs.BeamElement(length=2),
    bs.BeamElement(length=1),
]

nodes = [
    bs.BeamPinnedSupport(
        inbound_element=0,
        outbound_element=1,
        displacement=4,
    ),
]

boundary_conditions = [
    # W_0^0 = -1
    TMMEquation(
        # Format: (element_i, ElementEdge, state_var_i, coefficient)
        left_side=(0, ElementEdge.START, bs.BEAM_DISPLACEMENT, 1),
        right_side=-1,
    ),
    # W_e^1 = 2
    TMMEquation((1, ElementEdge.END, bs.BEAM_DISPLACEMENT, 1), 2),
    # M_0^0 = 0
    TMMEquation((0, ElementEdge.START, bs.BEAM_MOMENT, 1), 0),
    # M_e^1 = 0
    TMMEquation((1, ElementEdge.END, bs.BEAM_MOMENT, 1), 0),
]
```
</details>

The factory functions are supposed to shorten the code for common use cases.
If you want to combine different types of elements or supports, you can do this manually for small splines or define
a custom factory to populate them automatically.

Then the system of equations needs to be solved (`TMMSolver` class) to evaluate the beam state at any point (`BeamSpline` class):

```python
import numpy as np

import tmm_kit.beam_splines as bs
from tmm_kit import TMMSolver

solver = TMMSolver(bs.BEAM_STATE_SIZE, elements, nodes, bcs)
result = solver.solve()

spline = bs.BeamSpline(result, support_t)

# Evaluate an element at certain points...
element_states = spline.evaluate_element_state(
    element_i=1,
    s=[0.0, 0.5, 2.0],
)

# ...or the whole spline
spline_states = spline.evaluate_spline_state(
    t=np.linspace(support_t[0], support_t[-1], 128)
)
```

### Approximation / Smoothing

Another node and boundary conditions presets are used:

```python
import numpy as np

import tmm_kit.beam_splines as bs

# Data points are unevenly spaced
x = [2, 4, 5]
y = [-1, 4, 2]

# Use adaptive stiffness for step-independent smoothing
stiffness = bs.adaptive_stiffness(
    element_length=bs.centered_element_length(np.diff(x)),
    bandwidth=1,
)

# BeamPinnedSupport switched to BeamElasticSupport
elements, nodes = bs.create_beam_elements_and_nodes(
    support_t=x,
    element_class=bs.BeamElement,
    node_class=bs.BeamElasticSupport,
    node_data={
      "displacement": y[1:-1],
      "stiffness": stiffness[1:-1]
    },
)

# pinned_ends preset switched to elastic_ends
bcs = bs.elastic_ends(
    x[0], x[-1],
    stiffness[0], stiffness[-1],
)
```

## Corotational Beam Spline

Corotational beam spline uses the same core:
- Element classes
- Node classes
- Boundary conditions
- `TMMSolver`

The difference is:
- Special elements and nodes factory that computes `support_t` based on `polyline`.
  It also returns `angle_diffs` (angle misalignment array between polyline segments),
  which can be used for looped boundary conditions (more on this later).
- `CorotationalBeamSpline` evaluator that supports 2D point and curvature evaluation.

```python
import numpy as np

import tmm_kit.beam_splines as bs
from tmm_kit import TMMSolver

polyline = [[0, 0], [1, 1], [2, 1], [3, 2]]

elements, nodes, support_t, angle_diffs = bs.create_cbs_elements_and_nodes(
    polyline=polyline,
    element_class=bs.BeamElement,
    node_class=bs.BeamPinnedSupport,
)

bcs = bs.pinned_ends(0, 0)

solver = TMMSolver(bs.BEAM_STATE_SIZE, elements, nodes, bcs)
spline = bs.CorotationalBeamSpline(solver.solve(), support_t, polyline)

# You can evaluate beam states as for a usual beam spline...
eval_t = np.linspace(support_t[0], support_t[-1], 128)
states = spline.evaluate_spline_state(eval_t)

# ...as well as 2D coordinates of spline points...
points = spline.evaluate_spline_point(eval_t)

# ...or geometric curvatures
curvatures = spline.evaluate_spline_curvature(eval_t)
```

The iterative CBS polyline refinement is also supported (see `docs/beam_splines/corotational_beam_spline.md`
for more details):

```python
import tmm_kit.beam_splines as bs
from tmm_kit import TMMSolver

# Four circle points. The fifth closes the loop
circle_polyline = [[-2, 1], [2, 1], [2, -1], [-2, -1], [-2, 1]]

def set_up_solver(polyline, dummy_nodes_per_element):
    elements, nodes, support_t, angle_diffs = bs.create_cbs_elements_and_nodes(
        polyline=polyline,
        element_class=bs.BeamElement,
        node_class=bs.BeamPinnedSupport,
        dummy_nodes_per_element=dummy_nodes_per_element,
    )

    # Note the special loop boundary conditions that align the end directions
    bcs = bs.pinned_loop(0, angle_diffs[0])

    solver = TMMSolver(bs.BEAM_STATE_SIZE, elements, nodes, bcs)

    return solver, support_t


# Set up the first iteration
solver, support_t = set_up_solver(circle_polyline, 0)
spline = bs.CorotationalBeamSpline(solver.solve(), support_t, circle_polyline)

# Create a new polyline with 1 dummy node inserted per segment
refined_polyline = bs.refine_polyline(spline, 1)

# Set up the second iteration
solver, support_t = set_up_solver(refined_polyline, 1)
spline = bs.CorotationalBeamSpline(solver.solve(), support_t, refined_polyline)
```