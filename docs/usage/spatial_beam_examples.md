# Spatial Beam Code Examples

This document shows how to solve a 3D frame with the presets from `tmm_kit.spatial_beams`.
The model, the state variables and the node equations are described in `docs/math/spatial_beams/spatial_beam.md`.

## Branched Frame

A beam clamped at the start branches into two elements. The first branch turns by 60° and is free.
The second one goes straight, has a tangential load and a sliding support at its end.

Each element has its own stiffnesses and a local basis $(\vec{t}, \vec{n}, \vec{\beta})$ in global coordinates.
The basis is the identity by default (the element goes along the global $x$ axis);
`rotate_basis` turns a basis about one of its own axes:

```python
import numpy as np

import tmm_kit.spatial_beams as sb
from tmm_kit import ElementEdge, TMMSolver

# SE, GJ, EI_n, EI_beta
stiffness = (np.inf, 0.8, 1.0, 1.0)

elements = [
    sb.SpatialBeamElement(1.0, *stiffness),
    sb.SpatialBeamElement(
        1.0, *stiffness, basis=sb.rotate_basis(np.eye(3), sb.AXIS_B, np.pi / 3)
    ),
    sb.SpatialBeamElement(1.0, *stiffness, f_t=1.0),
]
```

A rigid node joins any number of inbound and outbound elements. It reads their bases from the element list.
The boundary conditions are attached to the remaining element edges, six equations per edge:

```python
nodes = [sb.SpatialBeamRigidNode([0], [1, 2], elements)]
bcs = [
    *sb.clamped_end(0, ElementEdge.START),
    *sb.free_end(1, ElementEdge.END),
    *sb.sliding_end(2, ElementEdge.END, sb.AXIS_N),
]

solver = TMMSolver(sb.SPATIAL_BEAM_STATE_SIZE, elements, nodes, bcs)
result = solver.solve()
```

The result holds the states in the local bases of the elements. Multiply a vector by the basis to get
its global coordinates:

```python
# Local states of the third element, shape (5, 12)
states = result.evaluate(2, np.linspace(0.0, 1.0, 5))

# Global displacement vectors, shape (5, 3)
displacements = (
    states[:, sb.SPATIAL_BEAM_DISPLACEMENT : sb.SPATIAL_BEAM_DISPLACEMENT + 3]
    @ elements[2].basis
)
```

In this problem only the axial force is non-zero: $N = -1$ in the first element and $N = s - 1$ in the third one.
