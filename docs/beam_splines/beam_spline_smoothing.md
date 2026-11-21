# Beam Spline for Smoothing (Approximation)

## Overview
Two methods of using a beam spline solved by a TMM (see `beam_spline.md`) for smoothing data. 

## Beam on Elastic Supports
The beam doesn't pass through node points $(t_i; \Pi_i)$, but instead it's considered to be connected to them
with springs (elastic supports).

It uses standard beam transfer equations with special conjugation equations that take into account the spring force:

1.  **Displacement is continuous:**
    $$ W_0^{i+1} - W_e^{i} = 0 $$
2.  **Slope is continuous:**
    $$ \theta_0^{i+1} - \theta_e^i = 0 $$
3.  **Moment is continuous:**
    $$ M_0^{i+1} - M_e^i = 0 $$
4.  **Force changes by the elastic force value:**
    $$ Q_0^{i+1} - Q_e^{i} = D_i (\Pi_{i+1} - W_0^{i+1}) $$

If the smoothed data points are not equally spaced (i.e. $L_i \neq L_j$), an adaptive support stiffness should be
used:

$$D_i = \frac{L_{i+1}}{g^4}$$

where $g$ is a bandwidth.

## Beam on Elastic Foundation

### Beam Model

The beam is not supported by discrete points, but rather is placed on a continuous elastic foundation.
It's based on a beam model proposed by Stephen Timoshenko for rails deformation analysis.
The model is described by these equations:

$$ \frac{dW}{dt} = \theta(t); \quad \frac{d\theta}{dt} = M(t); \quad \frac{dM}{dt} = Q(t); \quad \frac{dQ}{dt} = -\alpha W(t) + q(t);$$

For the smoothing application, we will consider the external load $q(t)$ to be caused by the foundation $Y(t)$ resistance:

$$ \frac{dQ}{dt} = \alpha (Y(t) - W(t)) = \frac{Y(t) - W(t)}{g^4}$$

Then previous equations can be combined into one differential equation:

$$ \frac{d^4W}{dt^4} + \frac{W(t)}{g^4} = \frac{Y(t)}{g^4}$$

In order to smooth data points $(t_i; \Pi_i)$, let the $Y(t)$ be a piecewise linear function.
In local coordinates ($l \in [0; L_i]$) it's formula is as follows:

$$Y(l) = \Pi_i + (\Pi_{i+1} - \Pi_i) \frac{l}{L_i}$$

The solution and the resulting transfer matrix are not documented here, because of their complexity.
We will say only that the transfer matrix takes into account the elastic force, so all four conjugation equations
must enforce continuity, unlike in the case of the beam on discrete elastic supports.

### Beam Response (Noise Attenuation Ratio)

Let's consider a periodic signal:

$$Y(t) = \Delta \cos{\frac{t}{\xi}}$$

If we substitute it into the differential equation of the beam model, we get this displacement:

$$W(t) = \Delta \frac{\xi^4}{g^4 + \xi^4} \cos{\frac{t}{\xi}}$$

Now we can find a ratio that describes the beam response to the signal of a certain frequency:

$$\varepsilon(\xi; g) = \frac{W(t)}{Y(t)} = \frac{\xi^4}{g^4 + \xi^4}$$
