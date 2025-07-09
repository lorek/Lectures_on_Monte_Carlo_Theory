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
## File: ch3_ellipsoid3D.py                                         ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng, PCG64

from matplotlib import cm
import argparse
import os


def ParseArguments():
    """
    Parse command-line arguments.
    --n_samples: total number of points used for correct sampling.
    --results_path: directory where figures will be saved.
    --a, --b, --c: semi-axes of the ellipsoid.
    --theta_precision: number of grid points for theta.
    --phi_precision: number of grid points for phi.
    """
    parser = argparse.ArgumentParser(description="Ellipsoid Simulation")
    parser.add_argument('--n_samples', default="1000", help='Total number of points (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Path for saving results (default: %(default)s)')
    parser.add_argument('--a', default="2", help='Semi-axis a (default: %(default)s)')
    parser.add_argument('--b', default="3", help='Semi-axis b (default: %(default)s)')
    parser.add_argument('--c', default="5", help='Semi-axis c (default: %(default)s)')
    parser.add_argument('--theta_precision', default="120", help='Number of grid points for theta (default: %(default)s)')
    parser.add_argument('--phi_precision', default="120", help='Number of grid points for phi (default: %(default)s)')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()
    return args

def heron(a, b, c):
    """Compute the area of a triangle using Heron's formula."""
    s = (a + b + c) / 2
    return (s * (s - a) * (s - b) * (s - c)) ** 0.5

def distance3d(x1, y1, z1, x2, y2, z2):
    """Compute the Euclidean distance between two 3D points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def areatriangle3d(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """Compute the area of a triangle in 3D."""
    a = distance3d(x1, y1, z1, x2, y2, z2)
    b = distance3d(x2, y2, z2, x3, y3, z3)
    c = distance3d(x3, y3, z3, x1, y1, z1)
    return heron(a, b, c)

def ellipsoid(theta, phi, a=2, b=3, c=5):
    """
    Map spherical coordinates (theta, phi) to an ellipsoid.
    theta in [0, 2pi], phi in [0, pi].
    """
    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi)
    return np.array([x, y, z])

def compute_distribution(theta_grid, phi_grid, a, b, c):
    """
    Compute the area-based distribution over a (theta, phi) grid.
    Returns the normalized distribution (2D array) and the total computed area.
    """
    # Compute ellipsoid coordinates on the grid.
    coords = ellipsoid(theta_grid, phi_grid, a, b, c)
    # Unpack grid coordinates.
    x_vals, y_vals, z_vals = coords
    n_theta = theta_grid.shape[0]
    n_phi = theta_grid.shape[1]
    distr = np.zeros((n_theta, n_phi))
    total_area = 0.0
    for i in range(n_theta - 1):
        for j in range(n_phi - 1):
            # Get the four corner points of the cell.
            x1, y1, z1 = x_vals[i, j],   y_vals[i, j],   z_vals[i, j]
            x2, y2, z2 = x_vals[i, j+1], y_vals[i, j+1], z_vals[i, j+1]
            x3, y3, z3 = x_vals[i+1, j], y_vals[i+1, j], z_vals[i+1, j]
            x4, y4, z4 = x_vals[i+1, j+1], y_vals[i+1, j+1], z_vals[i+1, j+1]
            # Divide cell into two triangles.
            tri1 = areatriangle3d(x1, y1, z1, x2, y2, z2, x3, y3, z3)
            tri2 = areatriangle3d(x4, y4, z4, x2, y2, z2, x3, y3, z3)
            cell_area = tri1 + tri2
            distr[i, j] = cell_area
            total_area += cell_area
    # Copy second-to-last column to last column to avoid issues.
    distr[:, -1] = distr[:, -2]
    # Normalize distribution so that sum equals 1.
    distr /= np.sum(distr)
    return distr, total_area

####################################
##             Main Section        ##
####################################

def main():
    # Parse arguments and assign variables.
    args = ParseArguments()
    n_samples = int(args.n_samples)

    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)


    a_val = float(args.a)
    b_val = float(args.b)
    c_val = float(args.c)
    theta_precision = int(args.theta_precision)
    phi_precision = int(args.phi_precision)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    domain_t = [0, 2 * np.pi]
    domain_u = [0, np.pi]
    t0, t1 = domain_t
    u0, u1 = domain_u

    # For display purposes.
    display = True

    ####################################
    ## Ellipsoid Points (3D Scatter - Regular Grid)
    ####################################
    theta_vals = np.linspace(0, 2*np.pi, theta_precision)
    phi_vals = np.linspace(0, np.pi, phi_precision)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals)
    coords_grid = ellipsoid(theta_grid, phi_grid, a=a_val, b=b_val, c=c_val)
    coords_grid_2d = coords_grid.reshape(3, theta_precision * phi_precision)

    if display:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect([1,1,1])
        mm_grid = np.min(coords_grid)
        MM_grid = np.max(coords_grid)
        ax.set_xlim(mm_grid, MM_grid)
        ax.set_ylim(mm_grid, MM_grid)
        ax.set_zlim(mm_grid, MM_grid)
        ax.scatter(coords_grid_2d[0, :], coords_grid_2d[1, :], coords_grid_2d[2, :],
                   s=1, color='blue')

        fig_path = os.path.join(results_path, "ellipsoid_scatter_regular_grid.pdf")
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
        fig.suptitle("Ellipsoid Points (Regular Grid)", fontsize=14)
        #plt.close(fig)

    view_configs = {
        "Side view": {
            "elev": 0, "azim": 0, "roll": 0,
            "filename_incorrect": "ch3_ellipsoid3D_points_incorrect_side_view.pdf",
            "filename_correct": "ch3_ellipsoid3D_points_correct_side_view.pdf"
        },
        "Top view": {
            "elev": 90, "azim": 0, "roll": 0,
            "filename_incorrect": "ch3_ellipsoid3D_points_incorrect_top_view.pdf",
            "filename_correct": "ch3_ellipsoid3D_points_correct_top_view.pdf"
        },
        "Under angle": {
            "elev": 16, "azim": -37, "roll": -12,
            "filename_incorrect": "ch3_ellipsoid3D_points_under_incorrect_angle.pdf",
            "filename_correct": "ch3_ellipsoid3D_points_under_correct_angle.pdf"
        }
    }


    #####################################################
    ##  Ellipsoid Points -- INCORECTLY sampled points ##
    #####################################################


    # Use PCG64
    rng = default_rng(PCG64(seed=31415))

    # "Incorrect" sampling: sample theta ~ U(0, 2pi) and phi ~ U(0, pi) independently.
    theta_incorrect = rng.uniform(0, 2*np.pi, n_samples)
    phi_incorrect   = rng.uniform(0, np.pi, n_samples)
    points_incorrect = ellipsoid(theta_incorrect, phi_incorrect, a=2, b=3, c=5)


    if display:


        # Compute limits based on the incorrectly sampled points.
        mm_incorrect = np.min(points_incorrect)
        MM_incorrect = np.max(points_incorrect)

        for view_name, params in view_configs.items():
            fig = plt.figure()
            #fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax_points = fig.add_subplot(projection='3d')
            ax_points.set_axis_off()
            # Set the view.
            ax_points.view_init(elev=params["elev"], azim=params["azim"], roll=params["roll"])
            # Set limits so the ellipsoid retains its proper shape.
            ax_points.set_xlim(mm_incorrect, MM_incorrect)
            ax_points.set_ylim(mm_incorrect, MM_incorrect)
            ax_points.set_zlim(mm_incorrect, MM_incorrect)
            # Plot the points.

            # Determine visible points based on the view:
            if view_name == "Top view":
                # Only show points on the upper half (z >= 0)
                visible = points_incorrect[2, :] >= 0
            elif view_name == "Side view":
                # Only show points on the front half (x >= 0)
                visible = points_incorrect[0, :] >= 0
            else:
                # For under angle, show all points.
                visible = np.full(points_incorrect.shape[1], True)

            ax_points.scatter(points_incorrect[0, visible],
                       points_incorrect[1, visible],
                       points_incorrect[2, visible],
                       s=10, color='blue', alpha=0.5)

            # if you want to display all points:
            # ax.scatter(points_incorrect[0, :], points_incorrect[1, :], points_incorrect[2, :], s=10, color='blue', alpha=0.5)

            fig_path = params["filename_incorrect"]
            plt.savefig(os.path.join(results_path, fig_path), bbox_inches='tight', pad_inches=0)
            fig.suptitle("Inorrectly sampled: "+view_name, fontsize=14)


    ####################################
    ## Ellipsoid Points -- CORRECTLY Sampled Points
    ####################################

    # Compute distribution over (theta, phi) using the grid.
    print("Computing area ...")
    distr, total_area = compute_distribution(theta_grid, phi_grid, a_val, b_val, c_val)
    distr_flat = distr.ravel()

    t, u = np.meshgrid(np.linspace(t0, t1, theta_precision),
                           np.linspace(u0, u1, phi_precision))


    # Build dictionary mapping each grid cell to its (theta, phi) pair.
    tu_dict = {}
    counter = 0
    for i in range(theta_grid.shape[0]):
        for j in range(theta_grid.shape[1]):
            tu_dict[counter] = [theta_grid[i, j], phi_grid[i, j]]
            counter += 1

    # Sample indices according to the computed distribution.
    sample_indices = np.random.choice(np.arange(len(distr_flat)), p=distr_flat, size=n_samples)

    # Retrieve sampled (theta, phi) values (optionally add a small offset)
    theta_correct = []
    phi_correct = []
    for idx in sample_indices:
        th, ph = tu_dict[idx]
        # Add a small random offset to avoid grid artifacts.
        th += (2*np.pi) * np.random.rand() / (theta_precision - 1)
        ph += np.pi * np.random.rand() / (phi_precision - 1)
        theta_correct.append(th)
        phi_correct.append(ph)

    # Compute final coordinates for correctly sampled points.
    coords_correct = ellipsoid(np.array(theta_correct), np.array(phi_correct), a=a_val, b=b_val, c=c_val)

    # Compute axis limits for the correctly sampled points.
    mm_correct = np.min(coords_correct)
    MM_correct = np.max(coords_correct)

    # Define view configurations.
    view_configs_correct = {
        "Side view": {"elev": 0, "azim": 0, "roll": 0,
                      "filename_correct": "ch3_ellipsoid3D_points_correct_side_view.pdf"},
        "Top view": {"elev": 90, "azim": 0, "roll": 0,
                     "filename_correct": "ch3_ellipsoid3D_points_correct_top_view.pdf"},
        "Under angle": {"elev": 16, "azim": -37, "roll": -12,
                        "filename_correct": "ch3_ellipsoid3D_points_correct_under_angle.pdf"}
    }

    print("Creating plots (takes a while) ...")
    if display:
        for view_name, params in view_configs_correct.items():
            fig = plt.figure(figsize=(7,7))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax_points = fig.add_subplot(projection='3d')
            ax_points.set_axis_off()
            ax_points.view_init(elev=params["elev"], azim=params["azim"], roll=params["roll"])
            ax_points.set_xlim(mm_correct, MM_correct)
            ax_points.set_ylim(mm_correct, MM_correct)
            ax_points.set_zlim(mm_correct, MM_correct)
            # Filter visible points based on view.
            if view_name == "Top view":
                visible = coords_correct[2, :] >= 0
            elif view_name == "Side view":
                visible = coords_correct[0, :] >= 0
            else:
                visible = np.full(coords_correct.shape[1], True)
            ax_points.scatter(coords_correct[0, visible],
                              coords_correct[1, visible],
                              coords_correct[2, visible],
                              s=10, color='blue', alpha=0.5)

            fig_path = os.path.join(results_path, params["filename_correct"])
            plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
            fig.suptitle("Correctly sampled: "+view_name, fontsize=14)
            #plt.close(fig)

    ####################################
    ## Ellipsoid -- Heatmap on Surface (Distribution)
    ####################################
    mm_grid = np.min(coords_grid)
    MM_grid = np.max(coords_grid)

    view_heatmap_configs = {
        "View A": {"elev": 31, "azim": 123, "roll": 175, "filename_heatmap": "ch3_ellipsoid3D_heatmap_A.pdf"},
        "View B": {"elev": 64, "azim": -145, "roll": -86, "filename_heatmap": "ch3_ellipsoid3D_heatmap_B.pdf"},
        "View C": {"elev": -16, "azim": 176, "roll": 180, "filename_heatmap": "ch3_ellipsoid3D_heatmap_C.pdf"}
    }

    if display:
        for view_name, params in view_heatmap_configs.items():
            fig = plt.figure(figsize=(5,5))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax_heatmap = fig.add_subplot(projection='3d')
            ax_heatmap.set_box_aspect([1,1,1])
            ax_heatmap.set_axis_off()
            ax_heatmap.view_init(elev=params["elev"], azim=params["azim"], roll=params["roll"])
            ax_heatmap.set_xlim(mm_grid, MM_grid)
            ax_heatmap.set_ylim(mm_grid, MM_grid)
            ax_heatmap.set_zlim(mm_grid, MM_grid)
            ax_heatmap.scatter(coords_grid_2d[0, :], coords_grid_2d[1, :], coords_grid_2d[2, :],
                               s=24, c=distr_flat)

            fig_path = os.path.join(results_path, params["filename_heatmap"])
            plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
            fig.suptitle("Heatmap: " + view_name, fontsize=14)
            #plt.close(fig)


    ####################################
    ##  Plot distribution surface (3D function)
    ####################################

    if display:
        # distribution grid
        fig_distr_surf = plt.figure(figsize=(7, 7))
        ax_distr_surf = fig_distr_surf.add_subplot(projection='3d')
        ax_distr_surf.view_init(elev=60, azim=-50, roll=15)
        # Normalize the distribution for coloring.
        norm_obj = plt.Normalize(distr.min(), distr.max())
        # Since distr is a 2D array, we use it directly.
        surf_colors = cm.viridis(norm_obj(distr))
        rcount, ccount = distr.shape
        ax_distr_surf.plot_surface(theta_grid, phi_grid, distr,
                                   rcount=rcount, ccount=ccount,
                                   facecolors=surf_colors, shade=False)
        ax_distr_surf.set_facecolor((0, 0, 0, 0))
        ax_distr_surf.xaxis.set_ticklabels([])
        ax_distr_surf.yaxis.set_ticklabels([])
        ax_distr_surf.zaxis.set_ticklabels([])

        fig_distr_surf_path = os.path.join(results_path, "ch3_ellipsoid3D_distr_3dfun.pdf")
        plt.savefig(fig_distr_surf_path, bbox_inches='tight', pad_inches=0)
        fig_distr_surf.suptitle("Distribution Surface", fontsize=14)



    ####################################
    ##  Plot distribution surface
    ####################################

    if display:
        # distribution grid
        fig_distr_grid = plt.figure()
        ax_distr_grid = fig_distr_grid.add_subplot()
        ax_distr_grid.imshow(distr, aspect=0.5)
        ax_distr_grid.set_axis_off()
        fig_distr_grid_path = os.path.join(results_path, "ch3_ellipsoid3D_distr_theta_phi.pdf")
        plt.savefig(fig_distr_grid_path,   bbox_inches='tight', pad_inches=0)
        fig_distr_grid.suptitle("Heatmap of distribution of parameters")

    ####################################
    ## Extra Figures for Sphere: x^2/2^2 + y^2/2^2 + z^2/2^2 = 1
    ####################################
    a_sphere, b_sphere, c_sphere = 2, 2, 2
    theta_grid_sphere, phi_grid_sphere = np.meshgrid(theta_vals, phi_vals)
    coords_sphere = ellipsoid(theta_grid_sphere, phi_grid_sphere, a=a_sphere, b=b_sphere, c=c_sphere)
    coords_sphere_2d = coords_sphere.reshape(3, theta_precision * phi_precision)
    distr_sphere, total_area_sphere = compute_distribution(theta_grid_sphere, phi_grid_sphere, a_sphere, b_sphere, c_sphere)
    distr_sphere_flat = distr_sphere.ravel()

    # a) Heatmap of parameters for sphere.
    fig_sphere_params = plt.figure()

    ax_sphere_params = fig_sphere_params.add_subplot()
    ax_sphere_params.imshow(distr_sphere, aspect=0.5)
    ax_sphere_params.set_axis_off()
    plt.savefig(os.path.join(results_path, "ch3_ellipsoid3D_sphere_heatmap_parameters.pdf"),
                bbox_inches='tight', pad_inches=0)
    fig_sphere_params.suptitle("Heatmap of Parameter Distribution (Sphere)", fontsize=14)



    # b) Heatmap on sphere surface.
    fig_sphere_surface = plt.figure(figsize=(4,4))
    ax_sphere_surface = fig_sphere_surface.add_subplot(projection='3d')
    ax_sphere_surface.view_init(elev=34, azim=-57, roll=4)
    ax_sphere_surface.set_axis_off()
    mm_sphere = np.min(coords_sphere)
    MM_sphere = np.max(coords_sphere)
    ax_sphere_surface.set_xlim(mm_sphere, MM_sphere)
    ax_sphere_surface.set_ylim(mm_sphere, MM_sphere)
    ax_sphere_surface.set_zlim(mm_sphere, MM_sphere)
    ax_sphere_surface.scatter(coords_sphere_2d[0, :],
                              coords_sphere_2d[1, :],
                              coords_sphere_2d[2, :],
                              s=30, c=distr_sphere_flat, cmap='viridis')

    plt.savefig(os.path.join(results_path, "ch3_ellipsoid3D_sphere_heatmap_on_surface.pdf"),
                bbox_inches='tight', pad_inches=0)
    fig_sphere_surface.suptitle("Heatmap on Sphere Surface", fontsize=14)

    # c) 3D surface plot of the sphere distribution.
    fig_sphere_surf = plt.figure(figsize=(8,4))
    ax_sphere_surf = fig_sphere_surf.add_subplot(projection='3d')
    ax_sphere_surf.view_init(elev=60, azim=-50, roll=15)
    norm_obj_sphere = plt.Normalize(distr_sphere.min(), distr_sphere.max())
    colors_sphere = cm.viridis(norm_obj_sphere(distr_sphere))
    rcount_s, ccount_s = distr_sphere.shape
    ax_sphere_surf.plot_surface(theta_grid_sphere, phi_grid_sphere, distr_sphere,
                                rcount=rcount_s, ccount=ccount_s,
                                facecolors=colors_sphere, shade=False)
    ax_sphere_surf.set_facecolor((0,0,0,0))
    ax_sphere_surf.xaxis.set_ticklabels([])
    ax_sphere_surf.yaxis.set_ticklabels([])
    ax_sphere_surf.zaxis.set_ticklabels([])
    plt.savefig(os.path.join(results_path, "ch3_ellipsoid3D_sphere_distribution_3dfun.pdf"),
                bbox_inches='tight', pad_inches=0)
    fig_sphere_surface.suptitle("Distribution Surface (Sphere)", fontsize=14)


    ####################################
    ## Final Summary
    ####################################
    print("Ellipsoid parameters: a =", a_val, ", b =", b_val, ", c =", c_val)

    #Knud Thomsen approximation:
    p = 1.6075
    thomsen_approx = 4 * np.pi * (((a_val * b_val)**p + (b_val * c_val)**p + (a_val * c_val)**p) / 3)**(1/p)

    print("Total area from patch-sum distribution:", total_area)
    print("Knud Thomsen approximation for ellipsoid area =", thomsen_approx)

    plt.show()

if __name__ == '__main__':
    main()