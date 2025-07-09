##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##
## Code for the book:                                               ##
## "Lectures on Monte Carlo Theory"                                 ##
## Authors: Paweł Lorek and Tomasz Rolski                           ##
## Affiliation: University of Wrocław                               ##
##                                                                  ##
## Publisher: Springer, 2025                                        ##
## Springer link: link.springer.com/book/9783032011893              ##
## Series: Probability Theory and  Stochastic Modelling             ##
## ISBN:  978-3-032-01189-3                                         ##
##                                                                  ##
## File: ch3_surface_area_ellipsoid.py                              ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
from scipy.integrate import dblquad

import numpy as np


def g_val(phi, theta, a, b, c):
    """
    Compute g(phi, theta) for the ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    parameterized by:
        x = a * sin(phi)*cos(theta)
        y = b * sin(phi)*sin(theta)
        z = c * cos(phi)
    with phi in [0, pi], theta in [0, 2*pi].

    This g(...) is the factor inside sqrt(...) for the surface area integral:
        Area = ∫∫ sqrt(g(phi,theta)) d(phi)d(theta).

    The expression is taken from your LaTeX:

    g(phi,theta) = sin^2(phi) * [
       c^2(a^2 - b^2) * cos^2(theta) * (cos^2(phi) - 1)
       + a^2( (b^2 - c^2)*cos^2(phi) + c^2 )
    ].
    """
    sinphi2 = np.sin(phi) ** 2
    cosphi2 = np.cos(phi) ** 2
    costheta2 = np.cos(theta) ** 2

    term1 = c ** 2 * (a ** 2 - b ** 2) * costheta2 * (cosphi2 - 1)
    term2 = a ** 2 * ((b ** 2 - c ** 2) * cosphi2 + c ** 2)

    return sinphi2 * (term1 + term2)


def ellipsoid_surface_area(a, b, c):
    """
    Numerically approximate the surface area of the ellipsoid
    x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    via the integral ∫∫ sqrt(g(phi,theta)) dphi dtheta.
    """

    def integrand(phi, theta):
        val = g_val(phi, theta, a, b, c)
        return np.sqrt(val) if val >= 0 else 0.0  # safeguard if numeric rounding goes negative

    area, error_est = dblquad(
        integrand,
        0.0, 2.0 * np.pi,  # theta range
        lambda theta: 0.0,  # phi lower bound
        lambda theta: np.pi  # phi upper bound
    )
    return area


# Example usage:
if __name__ == "__main__":
    a_val, b_val, c_val = 2.0, 3.0, 5.0
    area_approx = ellipsoid_surface_area(a_val, b_val, c_val)
    print("Approx ellipsoid area:", area_approx)


