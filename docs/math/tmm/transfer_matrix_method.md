# Transfer Matrix Method

Given a system of $N$ one-dimensional elements connected at their edges:
- $\vec{Y}^i(s)$ – state ($K \times 1$ vector) of an $i$-th element at a point $s \in [0;L_i]$.
- $\vec{Y}^i_0 = \vec{Y}^i(0)$ – state at the start of an $i$-th element.
- $\vec{Y}^i_e = \vec{Y}^i(L_i)$ – state at the end of an $i$-th element.

Each element has $K$ unknown state variables at each edge. The total number of unknowns: $2NK$.

In order to find these unknowns, a system of such equations must be solved:
1. **Transfer equations**: the relation between $\vec{Y}^i_e$ and $\vec{Y}^i_0$.
2. **Conjugation equations**: the relation between $\vec{Y}^{i+1}_0$ and $\vec{Y}^i_e$.
3. **Boundary conditions**: $K/2$ additional equations per each boundary element that make the number of equations equal the number of unknowns.

When we have state vectors $\vec{Y}^i_0$ at the start of each element, the transfer equations can be used to calculate a state vector at any point $\vec{Y}^i(s)$ of any element.

| Math                                | Code                                                        |
|-------------------------------------|-------------------------------------------------------------|
| Element, its length $L_i$           | `TMMElement`, `TMMElement.length`                           |
| Number of state variables $K$       | `state_size` argument of `TMMSolver`                        |
| Edge of an element (start or end)   | `ElementEdge.START`, `ElementEdge.END`                      |
| One equation of the system          | `TMMEquation`                                               |
| Assembling and solving the system   | `TMMSolver.solve`                                           |
| States $\vec{Y}^i_0$, $\vec{Y}^i_e$ | `TMMResult.get_edge_state`, `TMMResult.get_all_edge_states` |
| State $\vec{Y}^i(s)$                | `TMMResult.evaluate`                                        |

## Transfer Equations

The state of an element at a certain point can be calculated by the transfer equation:

$$\vec{Y}^i(s) = A(s)\vec{Y}^i_0+\vec{B}(s)$$

The transfer matrix $A(t)$ and the vector $\vec{B}(t)$ are found analytically by the Krylov method for each problem individually (see `krylov_method.md`).

The state at the end of the element is:

$$\vec{Y}^i_e = A(L_i)\vec{Y}^i_0+\vec{B}(L_i)$$

| Math                    | Code                                                    |
|-------------------------|---------------------------------------------------------|
| $A(s)$ and $\vec{B}(s)$ | `K` and `f` returned by `TMMElement.get_transfer_eq(s)` |

## Conjugation Equations

### Two Consequent Elements

Is used for beam splines and other systems without branching.

To pass the state from the end of an element to the beginning of the next one a conjugation equation is used:

$$\vec{Y}^{i+1}_0 = C\vec{Y}^i_e+\vec{D}$$

The simplest case is when the state is continuous: then $C = I$ (identity matrix) and $\vec{D} = 0$.

However, for some problems (such as a beam supported at conjugation points) the matrix form of this equation can't be used. Then the single matrix equation is replaced by a system of $K$ regular  equations.

| Math                                      | Code                         |
|-------------------------------------------|------------------------------|
| Node and its system of equations          | `TMMNode.get_conjugation_eq` |
| Continuity of a state variable            | `continuity_equation`        |
| Jump of a state variable by a given value | `offset_equation`            |

### Multiple Elements

Such cases are present in electrical circuits and pipeline modeling.

Let's say that the state consists of two groups of variables:
1. $I$ – Those that depend on direction (e.g., current or flow)
2. $E$ – Those that do not depend on direction (e.g., voltage or pressure)

For the first group:

$$\sum_1^k{I^i_e} + \hat{I} = \sum_{k+1}^K{I^i_0}$$

where $k$ is a number of inbound elements, $\hat{I}$ is an external factor.

For the second group:

$$E^1_e = E^2_e + \hat{E^2} = \dots = E^k_e + \hat{E^k} = E^{k+1}_0 + \hat{E^{k+1}} = \dots = E^{K}_0 + \hat{E^K}$$

where $\hat{E}$ is an external factor.

| Math                                                   | Code                                                    |
|--------------------------------------------------------|---------------------------------------------------------|
| Inbound and outbound elements of a node                | `TMMNode.inbound_elements`, `TMMNode.outbound_elements` |
| Electrical circuit: $E$ and $I$                        | `STATE_E`, `STATE_I` (`tmm_kit.electrical`)             |
| Electrical circuit node ($\hat{I} = 0$, $\hat{E} = 0$) | `ElectricalNode`                                        |
| Spatial beam rigid node                                | `SpatialBeamRigidNode` (`tmm_kit.spatial_beams`)        |
