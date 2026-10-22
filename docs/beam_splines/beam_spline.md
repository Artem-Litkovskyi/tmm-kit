# Beam Spline System via Transition Matrix Method

## Overview

**Beam spline** is a physics-based spline that is modeled as a beam, constrained by supports at its node points.

This document outlines a mathematical framework for solving Euler-Bernoulli beams using the **Transition Matrix Method (TMM)**. 

By preserving the state vectors at both edges of every segment (avoiding simplifications like the Three Moments Equation), this approach allows for the easy introduction of discontinuities, such as internal hinges or stepped displacements, later on.

## State Vector
The beam state at any cross-section is defined by the vector $\hat{Y}$:
$$ \hat{Y} = \begin{Bmatrix} W \\ \theta \\ M \\ Q \end{Bmatrix} $$
Where:
*   **$W$**: Displacement (directed along the y-axis)
*   **$\theta$**: Slope (rotation)
*   **$M$**: Bending moment
*   **$Q$**: Shear force

## Problem Geometry
*   The beam is divided into $N$ segments with edges placed at $N+1$ points at coordinates $t_i$.
*   Each segment $i$ has two edges, represented by states $\hat{Y}_0^i$ (start) and $\hat{Y}_e^i$ (end).
*   The length of the $i$-th segment is $L_i = t^{i+1} - t^i$.
*   This setup yields a total of **$2N$ state vectors**, or **$8N$ total unknowns**. To solve the system, we must construct exactly $8N$ linearly independent equations.

---

## 1. Standard Transition Equations (4N equations)
For each segment, the state at the end is related to the state at the start via the transition matrix $A(L_i)$:

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

After the beam is solved and all $\hat{Y}_0^i$ are known, the same transition matrix can be used to **evaluate** the beam state $W$ in any point.

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

---

## 3. Extra Boundary Conditions (4 equations)
To make the final system square ($8N$ equations for $8N$ unknowns), we must apply boundary conditions:

1.  **First node displacement:** $W_0^0 = \Pi_0$
2.  **Last node displacement:** $W_e^{N-1} = \Pi_N$
3.  **First node moment (e.g., pinned):** $M_0^0 = 0$
4.  **Last node moment (e.g., pinned):** $M_e^{N-1} = 0$

*(These can be swapped for clamped conditions, e.g., $\theta_0^0 = 0$, depending on the required boundary type).*

---

## Summary of the Global System
The final linear system takes the form of $K \cdot X = F$, where $K$ is an $8N \times 8N$ sparse matrix, $X$ is the vector of all $8N$ unknown states, and $F$ is the load/boundary vector.

| Equation Type         | Count    |
|:----------------------|:---------|
| Transition Equations  | $4N$     |
| Conjugation Equations | $4N - 4$ |
| Boundary Conditions   | $4$      |
| **Total Equations**   | **$8N$** |

With the shear constraints correctly relaxed at the nodes, the matrix $K$ will be non-singular and solvable via standard sparse linear algebra routines (e.g., `scipy.sparse.linalg.spsolve`).