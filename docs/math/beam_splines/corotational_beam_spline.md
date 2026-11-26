# Corotational Beam Spline

## Overview

**Corotational beam spline** is a geometric extension of the usual beam spline.

## Polyline
Each segment of the polyline represented by 2D points $\vec{A_i}$ has its local coordinate system:
*   **$\vec{t_i}$**: Tangent vector
*   **$\vec{n_i}$**: Normal vector

The **angle of misalignment** $\psi_i$ is an angle between two consecutive segments.

| Math                                                | Code                            |
|-----------------------------------------------------|---------------------------------|
| Angles of misalignment $\psi_i$                     | `angle_diff`                    |
| Elements and nodes of the polyline, $t_i$, $\psi_i$ | `create_cbs_elements_and_nodes` |

## Spline Evaluation
Let's assume that we have a beam state $\hat{Y_i}(s)$ at each local length coordinate $s \in [0; L_i]$.
Then the global position $\vec{R_i}(s)$ of a spline point:

$$ \vec{R_i}(s) = \vec{A_i} + \vec{t_i} \cdot s + \vec{n\theta_i}(s) \cdot W_i(s) $$

Where $\vec{n\theta_i}(s) = \vec{n_i} \cdot \cos{(\theta_i(s))} - \vec{t_i} \cdot \sin{(\theta_i(s))}$ is a rotated normal.
We need to rotate it to make edge normals identical $\vec{n\theta_i}(L_i) = \vec{n\theta_{i+1}}(0)$, so that the spline points
$\vec{R_i}(L_i) = \vec{R_{i+1}}(0)$ are continuous. Thus, the spline is called "corotational".

In other words:

$$ \vec{R_i}(s) = \vec{A_i} + \vec{t_i} \cdot X_i(s) + \vec{n_i} \cdot Y_i(s) $$

$$ X_i(s) = s - W_i(s) \cdot \sin{(\theta_i(s))}; \quad  Y_i(s) =  W_i(s) \cdot \cos{(\theta_i(s))} $$

Note that $\theta = \frac{dW}{ds}$ is a slope, so the calculations will be correct only for $\theta \to 0$, when it's close to the value of angle.

| Math                                | Element coordinate $s$                              | Spline coordinate $t$                              |
|-------------------------------------|-----------------------------------------------------|----------------------------------------------------|
| Point $\vec{R_i}(s)$                | `CorotationalBeamSpline.evaluate_element_point`     | `CorotationalBeamSpline.evaluate_spline_point`     |
| Rotated normal $\vec{n\theta_i}(s)$ | `CorotationalBeamSpline.evaluate_element_normal`    | `CorotationalBeamSpline.evaluate_spline_normal`    |
| Curvature of $\vec{R_i}(s)$         | `CorotationalBeamSpline.evaluate_element_curvature` | `CorotationalBeamSpline.evaluate_spline_curvature` |

## Conjugation Equations
To interpolate through polyline points, such conjugational equations are used:

$$ W_e^i = 0 $$

$$ W_0^{i+1} = 0 $$

$$ \theta_0^{i+1} - \theta_e^i = \psi_i $$

$$ M_0^{i+1} - M_e^i = 0 $$

The discontinuity of slope takes into account the angle of misalignment between polyline segments.

Every beam spline node has the `angle_misalignment` parameter for $\psi_i$, so the elastic supports
can be used to approximate the polyline points as well.

| Math                                             | Code                                                                   |
|--------------------------------------------------|------------------------------------------------------------------------|
| Interpolation node                               | `BeamPinnedSupport` with `displacement = 0`                            |
| Approximation node                               | `BeamElasticSupport`, `BeamElasticSupportInverted`                     |
| Closed polyline (the last node is the first one) | `pinned_loop`, `clamped_loop`, `elastic_loop`, `elastic_loop_inverted` |

## Accuracy Improvement with Dummy Nodes
The smoothest spline is achieved when $\psi_i \to 0$. When the polyline has sharp edges (e.g., a rectangle with
$\psi_i = \frac{\pi}{2}$) the result will have sharp breaks in the conjugating nodes. We can add more polyline points to avoid this.

One of the simplest options is to evaluate a point in the middle of each element $\vec{R_i}(\frac{L_i}{2})$ and insert it into the polyline.
Now $\psi$ is about two times smaller.

The new point wasn't present initially, so it shouldn't be pinned by $W = 0$. In fact it shouldn't affect the shape at all:

$$ W_0^{i+1} - W_e^i = 0 $$

$$ \theta_0^{i+1} - \theta_e^i = 0 $$

$$ M_0^{i+1} - M_e^i = 0 $$

$$ Q_0^{i+1} - Q_e^i = 0 $$

It's there just to split the segment into smaller ones with lower angle misalignment. Such nodes are called "dummy".

| Math                                             | Code               |
|--------------------------------------------------|--------------------|
| Dummy node                                       | `BeamDummyNode`    |
| Polyline with the inserted points                | `refine_polyline`  |
| Which points of the refined polyline are initial | `real_points_mask` |
