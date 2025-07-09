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
## File: ch3_ellipse_points.py                                      ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from numpy.random import default_rng, PCG64
from scipy.stats import norm
import scipy.special as ss
from scipy.special import ellipeinc
from pynverse import inversefunc

def ParseArguments():
    """Parse command-line arguments for the simulation."""
    parser = argparse.ArgumentParser(description="Ellipse Simulation and Inversion")
    parser.add_argument('--n_within', default="1000", help='Total number of points (default: %(default)s)')
    parser.add_argument('--n_wrong_grid', default="50", help='Total number of points (default: %(default)s)')
    parser.add_argument('--n_wrong_uniform', default="50", help='Total number of points (default: %(default)s)')
    parser.add_argument('--n_uniform', default="500", help='Total number of points (default: %(default)s)')
    parser.add_argument('--n_grid', default="100", help='Total number of points (default: %(default)s)')
    parser.add_argument('--n_cdf_grid', default="100", help='Total number of points (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Path for saving results (default: %(default)s)')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()
    return args



######################################
#            Main Code               #
######################################

def main():

    # Parse arguments
    args = ParseArguments()
    n_within = int(args.n_within)
    n_wrong_grid = int(args.n_wrong_grid)
    n_wrong_uniform = int(args.n_wrong_uniform)
    n_uniform = int(args.n_uniform)
    n_grid = int(args.n_grid)
    n_cdf_grid = int(args.n_cdf_grid)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)


    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    ########################################
    ##  Plot: Ellipse "On" (Incorrect)    ##
    ########################################

    # Generate   random angles and use R=1 (for the circle)
    theta = rng.uniform(0, 2*np.pi, n_wrong_uniform)
    theta_grid = np.linspace(0, 2*np.pi, n_wrong_grid)

    R_val = 1  # radius for the circle

    x_circle = R_val * np.cos(theta)
    y_circle = R_val * np.sin(theta)

    fig1 = plt.figure(figsize=(10, 2), dpi=360)
    ax1 = fig1.add_subplot(111)
    ax1.set_aspect('equal')
    ax1.scatter(x_circle, y_circle, s=20, facecolors='none', edgecolors='r')

    # Define transformation matrix for ellipse (a=5, b=1)
    a, b = 5, 1
    L = np.array([[a, 0], [0, b]])

    # Compute transformed points
    xy_circle = np.vstack((x_circle, y_circle))
    LX = L @ xy_circle
    ax1.scatter(LX[0, :], LX[1, :], s=20, facecolors='none', edgecolors='b')

    fig1_path = os.path.join(results_path, "ch3_ellipse_on_wrong_uniform.pdf")
    fig1.savefig(fig1_path, bbox_inches='tight', pad_inches=0)


    ##################################
    ##  Plot: Ellipse "On" (Grid)   ##
    ##################################

    fig2 = plt.figure(figsize=(10, 2), dpi=360)
    ax2 = fig2.add_subplot(111)
    ax2.set_aspect('equal')
    x_grid = np.cos(theta_grid)
    y_grid = np.sin(theta_grid)
    ax2.scatter(x_grid, y_grid, s=20, facecolors='none', edgecolors='r')

    xy_grid = np.vstack((x_grid, y_grid))
    LX_grid = L @ xy_grid
    ax2.scatter(LX_grid[0, :], LX_grid[1, :], s=20, facecolors='none', edgecolors='b')

    fig2_path = os.path.join(results_path, "ch3_ellipse_on_wrong_grid.pdf")
    fig2.savefig(fig2_path, bbox_inches='tight', pad_inches=0)


    ####################################
    ##  Plot: Points Within Ellipse   ##
    ####################################

    # Generate points uniformly within the circle using polar coordinates.

    theta_within = rng.uniform(0, 2*np.pi, n_within)
    R_within = np.sqrt(rng.random(n_within))  # sqrt for uniform density
    x_within = R_within * np.cos(theta_within)
    y_within = R_within * np.sin(theta_within)

    xy_within = np.vstack((x_within, y_within))
    LX_within = L @ xy_within

    fig3 = plt.figure(figsize=(10, 2), dpi=360)
    ax3 = fig3.add_subplot(111)
    ax3.scatter(x_within, y_within, s=1, color='r')
    ax3.scatter(LX_within[0, :], LX_within[1, :], s=1, facecolors='none', edgecolors='b')

    fig3_path = os.path.join(results_path, "ch3_ellipse_points_inside.pdf")

    fig3.savefig(fig3_path, bbox_inches='tight', pad_inches=0)


    ########################################
    ##  Plot: Inverse Function on Ellipse ##
    ########################################

    # Use a small number of points to illustrate the inverse function.

    theta_grid_inv = np.linspace(0, 2*np.pi, n_cdf_grid)
    m = -24
    F_vals = ellipeinc(theta_grid_inv, m) / ellipeinc(2*np.pi, m)

    fig4 = plt.figure(figsize=(10, 2), dpi=360)
    ax4 = fig4.add_subplot(111)
    ax4.plot(theta_grid_inv, F_vals, linewidth=1, color='k')

    # Define the function for inversion.
    F_fun = lambda x: ellipeinc(x, m) / ellipeinc(2*np.pi, m)
    # Generate 5 random uniform values.
    U_vals = rng.uniform(0, 1, 5)
    theta_inv = inversefunc(F_fun, y_values=U_vals)
    F_theta = ellipeinc(theta_inv, m) / ellipeinc(2*np.pi, m)

    for i in range(len(theta_inv)):
        ax4.plot([0, theta_inv[i], theta_inv[i]], [U_vals[i], U_vals[i], 0],
                 color='r', alpha=0.5, linewidth=0.5, linestyle='dashed')

    ax4.set_xlim(0, 2*np.pi)
    ax4.set_ylim(0, 1)
    fig4_path = os.path.join(results_path, "ch3_ellipse_inverse_uniform_cdf.pdf")
    fig4.savefig(fig4_path, bbox_inches='tight', pad_inches=0)


    ######################################################
    ##  Plot: Inverse Function with Uniform Ellipse     ##
    ######################################################


    F_fun = lambda w: ellipeinc(w, m) / ellipeinc(2*np.pi, m)
    U_uniform = rng.uniform(0, 1, n_uniform)
    theta_uniform = inversefunc(F_fun, y_values=U_uniform)

    fig5 = plt.figure(figsize=(10, 2), dpi=360)
    ax5 = fig5.add_subplot(111)
    ax5.scatter(a * np.cos(theta_uniform), b * np.sin(theta_uniform),
                s=2, facecolors='none', edgecolors='b')
    fig5_path = os.path.join(results_path, "ch3_ellipse_uniformly_sampled_points.pdf")
    fig5.savefig(fig5_path, bbox_inches='tight', pad_inches=0)


    n_samples_uniform = 100
    F_fun = lambda w: ellipeinc(w, m) / ellipeinc(2*np.pi, m)
    U_grid = np.linspace(0,1,n_grid)
    theta_uniform = inversefunc(F_fun, y_values=U_grid)

    fig5 = plt.figure(figsize=(10, 2), dpi=360)
    ax5 = fig5.add_subplot(111)
    ax5.scatter(a * np.cos(theta_uniform), b * np.sin(theta_uniform),
                s=2, facecolors='none', edgecolors='b')
    #ax5.set_xlim(0, 2*np.pi)
    #ax5.set_ylim(0, 1)
    fig5_path = os.path.join(results_path, "ch3_ellipse_grid_points.pdf")
    fig5.savefig(fig5_path, bbox_inches='tight', pad_inches=0)



    plt.show()

if __name__ == '__main__':
    main()
