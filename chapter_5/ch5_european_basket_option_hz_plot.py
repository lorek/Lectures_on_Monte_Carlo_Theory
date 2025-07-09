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
## File: ch5_european_basket_option_hz_plot.py                      ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
This script defines and plots a 2D function
    h(z1,z2) = (p1 * S1 * exp(mu1 + sigma1*z1) + p2 * S2 * exp(mu2 + sigma2*z2) - K)_+ * φ(z1,z2),
where φ(z1,z2) is the standard 2D normal density.
Default parameters are:
    S1 = 100, S2 = 55, K = 120,
    mu1 = 0.01875, sigma1 = 0.25,
    mu2 = 0.03,    sigma2 = 0.25,
    p1 = 0.4 (thus p2 = 0.6),
and a grid defined by zmin = -1, zmax = 4 with grid_points = 50.
The script creates a 3D plot of h(z1,z2) over the grid.
"""

import numpy as np
from scipy.stats import norm
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
from matplotlib import cm


def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Plot a 2D function h(z1,z2) for importance sampling in option pricing."
    )
    # Asset and option parameters
    parser.add_argument('--S1', type=float, default=100,
                        help="Initial asset price for asset 1 (default: %(default)s)")
    parser.add_argument('--S2', type=float, default=55,
                        help="Initial asset price for asset 2 (default: %(default)s)")
    parser.add_argument('--K', type=float, default=120,
                        help="Strike price (default: %(default)s)")

    # mu1, sigma1, mu2, sigma2
    parser.add_argument('--mu1', type=float, default=0.01875,
                        help="Drift for asset 1 (default: %(default)s)")
    parser.add_argument('--sigma1', type=float, default=0.25,
                        help="Volatility for asset 1 (default: %(default)s)")
    parser.add_argument('--mu2', type=float, default=0.03,
                        help="Drift for asset 2 (default: %(default)s)")
    parser.add_argument('--sigma2', type=float, default=0.25,
                        help="Volatility for asset 2 (default: %(default)s)")

    # Mixing weight
    parser.add_argument('--p1', type=float, default=0.4,
                        help="Weight p1 (default: %(default)s)")

    # Grid parameters for plotting
    parser.add_argument('--zmin', type=float, default=-1,
                        help="Minimum z value for grid (default: %(default)s)")
    parser.add_argument('--zmax', type=float, default=4,
                        help="Maximum z value for grid (default: %(default)s)")
    parser.add_argument('--grid_points', type=int, default=50,
                        help="Number of grid points in each dimension (default: %(default)s)")

    # Last two entries: seed and results_path
    parser.add_argument('--seed', type=str, default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', type=str, default="results",
                        help="Folder for saving output figures (default: %(default)s)")
    return parser.parse_args()


def h_2d(z1, z2, S1, S2, K, mu1, sigma1, mu2, sigma2, p1, p2):
    """
    Computes the function
      h(z1,z2) = (p1 * S1 * exp(mu1 + sigma1*z1) + p2 * S2 * exp(mu2 + sigma2*z2) - K)_+ * φ(z1,z2),
    where φ(z1,z2) = (1/(2π)) * exp(-0.5*(z1^2+z2^2)).
    """
    term = p1 * S1 * np.exp(mu1 + sigma1 * z1) + p2 * S2 * np.exp(mu2 + sigma2 * z2) - K
    payoff = np.maximum(term, 0)
    phi_2d = 1 / (2 * np.pi) * np.exp(-0.5 * (z1 ** 2 + z2 ** 2))
    return payoff * phi_2d


def main():
    args = ParseArguments()
    os.makedirs(args.results_path, exist_ok=True)
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(seed)

    # Extract parameters into local variables
    S1 = args.S1
    S2 = args.S2
    K = args.K
    mu1 = args.mu1
    sigma1 = args.sigma1
    mu2 = args.mu2
    sigma2 = args.sigma2
    p1 = args.p1
    p2 = 1 - p1
    zmin = args.zmin
    zmax = args.zmax
    grid_points = args.grid_points

    # Display a brief summary of what the code is doing and the parameters being used.
    print("Plotting the 2D function:")
    print("  h(z1,z2) = (p1 * S1 * exp(mu1 + sigma1*z1) + p2 * S2 * exp(mu2 + sigma2*z2) - K)_+ * φ(z1,z2)")
    print("\nParameters:")
    print(f"  S1 = {S1}, S2 = {S2}, K = {K}")
    print(f"  mu1 = {mu1}, sigma1 = {sigma1}")
    print(f"  mu2 = {mu2}, sigma2 = {sigma2}")
    print(f"  p1 = {p1}, p2 = {p2}")
    print(f"  Grid: z1,z2 ∈ [{zmin}, {zmax}] with {grid_points} points per axis\n")

    # Create a grid for z1 and z2
    z1_vals = np.linspace(zmin, zmax, grid_points)
    z2_vals = np.linspace(zmin, zmax, grid_points)
    Z1, Z2 = np.meshgrid(z1_vals, z2_vals)

    # Evaluate h(z1,z2) on the grid
    H = h_2d(Z1, Z2, S1, S2, K, mu1, sigma1, mu2, sigma2, p1, p2)

    # Create a surface plot with white faces and black edges (opaque)
    fig2 = plt.figure(figsize=(9,9))
    ax2 = fig2.add_subplot(111, projection='3d')
    surf2 = ax2.plot_surface(
        Z1, Z2, H,
        color='white',    # fill color
        edgecolor='black',  # wireframe color
        linewidth=0.5,
        alpha=1.0,        # fully opaque
        shade=False       # flat color
    )

    # Turn off the default Matplotlib grid.
    ax2.grid(False)

    # Label the axes
    ax2.set_xlabel("$z_1$")
    ax2.set_ylabel("$z_2$")
    #ax2.set_zlabel("$h(z_1,z_2)$")

    # Make the background planes transparent (only axis lines visible)
    ax2.xaxis.pane.set_alpha(0.0)
    ax2.yaxis.pane.set_alpha(0.0)
    ax2.zaxis.pane.set_alpha(0.0)
    ax2.xaxis.pane.set_edgecolor('black')
    ax2.yaxis.pane.set_edgecolor('black')
    ax2.zaxis.pane.set_edgecolor('black')
    ax2.view_init(elev=35, azim=-80, roll=0)


    output_file2 = os.path.join(args.results_path, "ch5_european_basket_option_hz_plot.pdf")
    fig2.savefig(output_file2, format="pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
