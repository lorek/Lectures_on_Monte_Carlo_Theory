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
## File: ch4_human_height_weight_conf_regions.py                    ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
This script reads a CSV file containing height and weight data, computes summary statistics
and confidence intervals (using the notations hat{Y}^1_R for weight and hat{Y}^2_R for height),
and then computes a 95% confidence ellipse and parallelogram. Finally, it plots the data along with
the confidence regions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import matplotlib.transforms as transforms
import argparse
import os
from tabulate import tabulate
from scipy.stats import chi2
from scipy.linalg import sqrtm
from scipy.stats import norm


def ParseArguments():
    parser = argparse.ArgumentParser(description="Confidence regions for height and weight data.")
    parser.add_argument('--input_file', default="human_height_weight_small.csv",
                        help="CSV file with data (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving outputs (default: %(default)s)")
    args = parser.parse_args()
    return args


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Plot the covariance confidence ellipse of x and y on the provided axis.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def main():
    # Parse arguments and create output folder.
    args = ParseArguments()

    input_file = args.input_file
    results_path = args.results_path

    os.makedirs(results_path, exist_ok=True)

    # Read data from CSV (assumes separator is ";" as before)
    data = pd.read_csv(input_file, sep=";")
    # Original CSV columns: "height" and "weight"
    # For our new notation:
    #   Let \(\hat{Y}^1_R\) denote the sample mean of weight,
    #   and \(\hat{Y}^2_R\) the sample mean of height.
    weights = data["weight"].to_numpy()  # will be on x-axis (Y^1_R)
    heights = data["height"].to_numpy()  # will be on y-axis (Y^2_R)
    R = len(weights)

    # Compute sample means and standard deviations.
    Y1_mean = np.mean(weights)  # hat{Y}^1_R
    Y1_std = np.std(weights, ddof=1)  # hat{S}^1_R
    Y2_mean = np.mean(heights)  # hat{Y}^2_R
    Y2_std = np.std(heights, ddof=1)  # hat{S}^2_R

    # 95% confidence intervals using z=1.96
    z_val = 1.96
    CI_Y1 = (Y1_mean - z_val * Y1_std / np.sqrt(R), Y1_mean + z_val * Y1_std / np.sqrt(R))
    CI_Y2 = (Y2_mean - z_val * Y2_std / np.sqrt(R), Y2_mean + z_val * Y2_std / np.sqrt(R))

    # Create summary table for means, stds, and confidence intervals.
    table_stats = [
        ["", "Mean", "Std Dev", "95% CI"],
        [r"Weight (hat{Y}^1_R)", f"{Y1_mean:.4f}", f"{Y1_std:.4f}", f"({CI_Y1[0]:.4f}, {CI_Y1[1]:.4f})"],
        [r"Height (hat{Y}^2_R)", f"{Y2_mean:.4f}", f"{Y2_std:.4f}", f"({CI_Y2[0]:.4f}, {CI_Y2[1]:.4f})"]
    ]
    print("\n--- Summary Statistics ---")
    print(tabulate(table_stats, headers="firstrow", tablefmt="grid"))

    # Compute covariance matrix based on [weight, height] (order matters).
    S_R_square = np.cov(np.vstack((weights, heights)), ddof=1)
    print("\nCovariance matrix S_R2 (Weight, Height):")
    print(S_R_square)

    # Compute the parallelogram corners using sqrtm.
    S_R = sqrtm(S_R_square)  #  full symmetric square root of a matrix such that  S_R S_R = S_R_square

    # Eigen-decomposition of the covariance matrix.
    eig_vals, eig_vecs = np.linalg.eig(S_R_square)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    print("\nEigenvalues: [lambda_1, lambda_2]=")
    print(eig_vals)
    print("\nEigenvectors (columns correspond to eigenvalues): [f_1, f_2]=")
    print(eig_vecs)

    # Compute 95% quantile for chi-square with 2 df and corresponding z value.
    from scipy.stats import chi2
    chi2_2_95 = chi2.ppf(0.95, df=2)

    z_2d_95 = norm.ppf((np.sqrt(0.95) + 1) / 2)
    print(f"\nChi-square quantile (df=2, 0.95): {chi2_2_95:.4f}")
    print(f"Corresponding z (2D): {z_2d_95:.5f}")

    # Compute principal axes for the confidence ellipse:
    g1 = np.sqrt((chi2_2_95 * eig_vals[0]) / R) * eig_vecs[:, 0]
    g2 = np.sqrt((chi2_2_95 * eig_vals[1]) / R) * eig_vecs[:, 1]
    print("\nPrincipal axes:")
    print("g1 =", g1)
    print("g2 =", g2)

    print("\nMatrix S_R such that S_R S_R = S_R2")
    print("\nS_R = ", S_R)

    # Instead of using a fixed value for z, compute z_2d_95 as above.
    # The corners of the hypercube [-z, z]^2 are:
    v_corners = np.array([
        [-z_2d_95, -z_2d_95],
        [-z_2d_95, z_2d_95],
        [z_2d_95, z_2d_95],
        [z_2d_95, -z_2d_95]
    ])
    w_corners = []
    for v in v_corners:
        w = (1 / np.sqrt(R)) * S_R @ v + np.array([Y1_mean, Y2_mean])
        w_corners.append(w)
    w_corners = np.array(w_corners)

    # Display parallelogram corners in a table.
    table_par = []
    for i, corner in enumerate(w_corners):
        table_par.append([f"w_{i + 1}", f"({corner[0]:.4f}, {corner[1]:.4f})"])
    print("\n--- Parallelogram Corners ---")
    print(tabulate(table_par, headers=["Corner", "Coordinates"], tablefmt="grid"))



    # For ellipse patch, use:
    major_len = 2.0 * np.sqrt(eig_vals[0] * chi2_2_95 / R)
    minor_len = 2.0 * np.sqrt(eig_vals[1] * chi2_2_95 / R)
    angle_deg = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
    ellipse_patch = Ellipse(xy=(Y1_mean, Y2_mean),
                            width=major_len, height=minor_len,
                            angle=angle_deg,
                            edgecolor='black', facecolor='gray', alpha=0.6, lw=2, label="Confidence Ellipse")
    ellipse_patch2 = Ellipse(xy=(Y1_mean, Y2_mean),
                            width=major_len, height=minor_len,
                            angle=angle_deg,
                            edgecolor='black', facecolor='gray', alpha=0.6, lw=2, label="Confidence Ellipse")

    # For parallelogram  use:
    parallelogram = Polygon(w_corners, closed=True, edgecolor='brown', linewidth=2, facecolor='none')
    parallelogram2 = Polygon(w_corners, closed=True, edgecolor='brown', linewidth=2, facecolor='none')


    # First plot:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Define axis limits (hand-crafted, for good visibility)
    xmin, xmax = 107.4817, 147.4817
    ymin, ymax = 61, 74
    ax.set_xlim(xmin, xmax)

    ax.set_ylim(ymin, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', xmin))
    ax.spines['bottom'].set_position(('data', ymin))

    # Scatter plot
    ax.scatter(data["weight"], data["height"], s=2, color='black', alpha=0.7)
    ax.add_patch(ellipse_patch)
    ax.add_patch(parallelogram)

    ax.set_xlabel("weight", fontsize=12)
    ax.set_ylabel("height", fontsize=12)
    ax.xaxis.set_label_coords(0.95, 0.05)
    ax.yaxis.set_label_coords(0.0275, 0.93)


    # save
    plt.savefig(os.path.join(results_path, "ch4_human_height_weight_ellipse_parall_1.pdf"),
                bbox_inches='tight', pad_inches=0)



    #
    # # Second plot:
    # Second plot: Detailed view using computed confidence intervals and principal axes
    xmin2, xmax2 = 126.5, 128.5
    ymin2, ymax2 = 67, 68.9875



    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(xmin2, xmax2)
    ax.set_ylim(ymin2, ymax2)


    # Optionally reposition the spines (here we let them be at the computed limits)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('data', xmin2))
    ax.spines['bottom'].set_position(('data', ymin2))

    ax.set_xticks([CI_Y1[0], Y1_mean, CI_Y1[1]])
    ax.set_yticks([CI_Y2[0], Y2_mean, CI_Y2[1]])

    # Scatter plot of the data points (using the original data)
    ax.scatter(data["weight"], data["height"], s=10, color='black', alpha=0.5)

    # Draw dashed reference lines corresponding to the confidence intervals
    ax.axvline(x=CI_Y1[0], linestyle='--', color='gray', alpha=0.5)
    ax.axvline(x=CI_Y1[1], linestyle='--', color='gray', alpha=0.5)
    ax.axhline(y=CI_Y2[0], linestyle='--', color='gray', alpha=0.5)
    ax.axhline(y=CI_Y2[1], linestyle='--', color='gray', alpha=0.5)

    # Draw dashed lines from the center to the left (x = CI_Y1[0]) and bottom (y = CI_Y2[0]) boundaries
    ax.plot([0, Y1_mean], [Y2_mean, Y2_mean], linestyle='-', color='gray', alpha=0.5)
    ax.plot([Y1_mean, Y1_mean], [Y2_mean, 0], linestyle='-', color='gray', alpha=0.5)

    # Add the previously computed confidence ellipse (ellipse_patch2)
    ax.add_patch(ellipse_patch2)

    # Draw arrows representing the principal axes (g1 and g2) starting from (Y1_mean, Y2_mean)
    origin = np.array([Y1_mean, Y2_mean])
    ax.arrow(origin[0], origin[1], g1[0], g1[1],
             head_width=0.02, head_length=0.04, fc='black', ec='black',
             lw=1, length_includes_head=True, label="Principal axis 1")
    ax.arrow(origin[0], origin[1], g2[0], g2[1],
             head_width=0.02, head_length=0.04, fc='black', ec='black',
             lw=1, length_includes_head=True, label="Principal axis 2")


    # Mark the endpoints of the principal axes with black dots
    ax.scatter([origin[0] + g1[0], origin[0] + g2[0]],
               [origin[1] + g1[1], origin[1] + g2[1]],
               color='black', s=20)

    # Add the confidence parallelogram (parallelogram2)
    ax.add_patch(parallelogram2)

    ax.set_xlabel("Weight", fontsize=12)
    ax.set_ylabel("Height", fontsize=12)
    ax.xaxis.set_label_coords(0.95, 0.05)
    ax.yaxis.set_label_coords(0.025, 0.93)

    plt.savefig(os.path.join(results_path, "ch4_human_height_weight_ellipse_parall_2.pdf"),
                bbox_inches='tight', pad_inches=0)




    plt.show()


if __name__ == "__main__":
    main()
