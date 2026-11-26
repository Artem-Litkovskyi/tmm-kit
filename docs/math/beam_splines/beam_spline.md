# Beam Spline System via Transfer Matrix Method

## Overview

**Beam spline** is a physics-based spline that is modeled as a beam, constrained by supports at its node points.

This document outlines a mathematical framework for solving Euler-Bernoulli beams using the **Transfer Matrix Method (TMM)**. 

By preserving the state vectors at both edges of every segment (avoiding simplifications like the Three Moments Equation), this approach allows for the easy introduction of discontinuities, such as stepped displacements, later on.

## State Vector
The beam state at any cross-section is defined by the vector $\hat{Y}$:
$$ \hat{Y} = \begin{Bmatrix} W \\ \theta \\ M \\ Q \end{Bmatrix} $$
Where:

| Variable                 | Meaning                                  | Constant            |
|--------------------------|------------------------------------------|---------------------|
| $W$                      | Displacement (directed along the y-axis) | `BEAM_DISPLACEMENT` |
| $\theta = \frac{dW}{dt}$ | Slope                                    | `BEAM_SLOPE`        |
| $M = \frac{d\theta}{dt}$ | Bending moment                           | `BEAM_MOMENT`       |
| $Q = \frac{dM}{dt}$      | Shear force                              | `BEAM_FORCE`        |

The variables are stored in this order, `BEAM_STATE_SIZE = 4`.

## Problem Geometry
*   The beam is divided into $N$ segments with edges placed at $N+1$ points at coordinates $t_i$.
*   Each segment $i$ has two edges, represented by states $\hat{Y}_0^i$ (start) and $\hat{Y}_e^i$ (end).
*   The length of the $i$-th segment is $L_i = t^{i+1} - t^i$.
*   This setup yields a total of **$2N$ state vectors**, or **$8N$ total unknowns**. To solve the system, we must construct exactly $8N$ linearly independent equations.

---

## 1. Standard Transfer Equations (4N equations)
For each segment, the state at the end is related to the state at the start via the transfer matrix $A(L_i)$:

$$ \hat{Y}_e^i = A(L_i) \cdot \hat{Y}_0^i $$

When rearranged into the global matrix system as $\hat{Y}_e^i - A(L_i) \cdot \hat{Y}_0^i = 0$, this provides **4 equations per segment** (total of $4N$ equations).

For a standard Euler-Bernoulli beam with constant stiffness $EI$, the matrix $A(s)$ is:
$$ A(t) = \begin{bmatrix} 
1 & s & \frac{s^2}{2EI} & \frac{s^3}{6EI} \\ 
0 & 1 & \frac{s}{EI} & \frac{s^2}{2EI} \\ 
0 & 0 & 1 & s \\ 
0 & 0 & 0 & 1 
\end{bmatrix} $$

where $s \in [0; L_i]$ is a local coordinate.

After the beam is solved and all $\hat{Y}_0^i$ are known, the same transfer matrix can be used to **evaluate** the beam state $W$ in any point.

| Element                                    | Class                   |
|--------------------------------------------|-------------------------|
| Euler-Bernoulli beam with $EI = 1$         | `BeamElement`           |
| The same beam stretched by an axial force  | `BeamElementStretched`  |
| The same beam compressed by an axial force | `BeamElementCompressed` |

| Evaluation                                       | Method                              |
|--------------------------------------------------|-------------------------------------|
| State at a local coordinate $s$ of an element    | `BeamSpline.evaluate_element_state` |
| State at a spline coordinate $t$                 | `BeamSpline.evaluate_spline_state`  |
| States $\hat{Y}_0^i$, $\hat{Y}_e^i$ at the edges | `BeamSpline.get_edge_state`         |

---

## 2. Standard Conjugation Equations (4N - 4 equations)
At each of the $N-1$ nodes, the end of the $i$-th segment meets the start of the $(i+1)$-th segment. To interpolate points $(t_i; \Pi_i)$ we construct such equations:

1.  **End of previous segment hits the node:**
    $$ W_e^i = \Pi_{i+1} $$
2.  **Start of next segment hits the node:**
    $$ W_0^{i+1} = \Pi_{i+1} $$
3.  **Slope is continuous across the node:**
    $$ \theta_0^{i+1} - \theta_e^i = 0 $$
4.  **Moment is continuous across the node:**
    $$ M_0^{i+1} - M_e^i = 0 $$

*Note: The shear forces $Q_e^i$ and $Q_0^{i+1}$ are intentionally uncoupled. The difference between them after the matrix is solved will naturally equal the vertical reaction force of the node.*

This provides **4 equations per internal node**, totaling $4(N-1) = 4N - 4$ equations.

| Node                                 | Class               | Parameters                   |
|--------------------------------------|---------------------|------------------------------|
| Support that the beam passes through | `BeamPinnedSupport` | `displacement` – $\Pi_{i+1}$ |

---

## 3. Extra Boundary Conditions (4 equations)
To make the final system square ($8N$ equations for $8N$ unknowns), we must apply boundary conditions:

1.  **First node displacement:** $W_0^0 = \Pi_0$
2.  **Last node displacement:** $W_e^{N-1} = \Pi_N$
3.  **First node moment (e.g., pinned):** $M_0^0 = 0$
4.  **Last node moment (e.g., pinned):** $M_e^{N-1} = 0$

*(These can be swapped for clamped conditions, e.g., $\theta_0^0 = 0$, depending on the required boundary type).*

| Boundary type         | Function       | Equations at the first and the last nodes                         |
|-----------------------|----------------|-------------------------------------------------------------------|
| Pinned                | `pinned_ends`  | Set displacements, zero moments                                   |
| Clamped               | `clamped_ends` | Set displacements and slopes                                      |
| Free                  | `free_ends`    | Zero moments and forces                                           |
| Pinned, closed curve  | `pinned_loop`  | Set displacement at the closing node, continuous slope and moment |
| Clamped, closed curve | `clamped_loop` | Set displacement and slope at the closing node                    |

---

## Summary of the Global System
The final linear system takes the form of $K \cdot X = F$, where $K$ is an $8N \times 8N$ sparse matrix, $X$ is the vector of all $8N$ unknown states, and $F$ is the load/boundary vector.

| Equation Type         | Count    |
|:----------------------|:---------|
| Transfer Equations    | $4N$     |
| Conjugation Equations | $4N - 4$ |
| Boundary Conditions   | $4$      |
| **Total Equations**   | **$8N$** |

With the shear constraints correctly relaxed at the nodes, the matrix $K$ will be non-singular and solvable via standard sparse linear algebra routines (e.g., `scipy.sparse.linalg.spsolve`).

The elements and nodes of a spline are populated by `create_beam_elements_and_nodes`,
the system is assembled and solved by `TMMSolver`.
