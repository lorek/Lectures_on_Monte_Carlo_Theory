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
## File: ch5_BrownianMotion_bisection_stratified.py                 ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Stratified Sampling of Brownian Motion via the Bisection Method

This script simulates a Brownian motion using stratified sampling to generate
the two key increments used in the bisection method. The code illustrates the
stratification of two independent standard normal variables (used for B(1/2)
and B(1)) via specified strata.

The simulation parameters (including the PRNG seed and output folder) can be
set via command-line arguments.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import norm
import argparse
import os

def ParseArguments():
    parser = argparse.ArgumentParser(description="Stratified Sampling of Brownian Motion")
    parser.add_argument('--nr_stratas', type=int, default=5,
                        help="Number of strata to use in the stratification of the normal distribution (default: %(default)s)")
    parser.add_argument('--strata05', type=int, default=4,
                        help="Stratum index to sample for the B(1/2) increment (default: %(default)s)")
    parser.add_argument('--strata1', type=int, default=2,
                        help="Stratum index to sample for the B(1) increment (default: %(default)s)")
    parser.add_argument('--k', type=int, default=10,
                        help="Parameter k: the level of dyadic partition for the Brownian path (default: %(default)s)")
    parser.add_argument('--max_iter', type=int, default=10,
                        help="Number of replicates for the stratified sampling (default: %(default)s)")
    parser.add_argument('--seed', type=str, default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed")
    parser.add_argument('--results_path', type=str, default="results",
                        help="Folder where the output figures (PDF) will be saved")
    return parser.parse_args()


def main():
    # Parse command-line arguments.
    args = ParseArguments()
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(seed)
    os.makedirs(args.results_path, exist_ok=True)

    # Simulation parameters
    nr_stratas = args.nr_stratas

    strata05 = args.strata05
    strata1 = args.strata1
    # Validate that strata indices are within [0, nr_stratas-1]
    if not (1 <= strata05 <= nr_stratas):
        raise ValueError("strata05 must be in the {1,2,...,nr_stratas}.")
    if not (1 <= args.strata1 <= nr_stratas):
        raise ValueError("strata1 must be in the interval {1,2,...,nr_stratas}.")

    # Adjust numbering: in book we enumerate stratas 1,2,..,nr_stratas, below in code 0,..,nr_stratas-1
    strata05 = args.strata05 - 1
    strata1 = args.strata1 - 1

    k = args.k              # determines the time grid: 2**k+1 points in [0,1]
    max_iter = args.max_iter

    # Define the strata for a standard normal: we use m-1 quantiles
    p = np.arange(1, nr_stratas) / nr_stratas
    stratas = norm.ppf(p)
    print("p =", p)
    print("Strata boundaries (quantiles):", stratas)
    print(f"Strata for B(1/2): [{strata05/nr_stratas:.3f}, {(strata05+1)/nr_stratas:.3f}] -> "
          f"Z in [{norm.ppf(strata05/nr_stratas):.3f}, {norm.ppf((strata05+1)/nr_stratas):.3f}]")
    print(f"Strata for B(1):   [{strata1/nr_stratas:.3f}, {(strata1+1)/nr_stratas:.3f}] -> "
          f"Z in [{norm.ppf(strata1/nr_stratas):.3f}, {norm.ppf((strata1+1)/nr_stratas):.3f}]")

    # Compute y-axis limits for plotting (based on strata)
    y_min = 2 * np.min(stratas)
    y_max = 2 * np.max(stratas)

    # Create an array of strata boundaries for plotting rectangles.
    stratas_ys = np.append(np.insert(stratas, 0, y_min), y_max)
    rec_width = 0.04

    # ---------------------------------------------
    # Generate stratified samples for Z(2^(k-1)) and Z(2^k)
    # ---------------------------------------------
    V05 = strata05 / nr_stratas + 1 / nr_stratas * rng.random(max_iter)
    V1  = strata1  / nr_stratas + 1 / nr_stratas * rng.random(max_iter)
    Z05 = norm.ppf(V05)
    Z1  = norm.ppf(V1)

    # ---------------------------------------------
    # Plot 1: Strata Rectangles and Stratified Samples (Z05 and Z1)
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlim(-0.05, 1.05)
    # Draw the strata as colored rectangles
    color_list = ["#e1dae6", "#bfbfbf", "#848186", "#d0d0d0", "#a9a4ad"]
    for i in reversed(range(len(stratas_ys) - 1)):
        # Rectangle for the stratum at x=0.5
        ax.add_patch(Rectangle((0.5 - rec_width/2, stratas_ys[i]),
                               rec_width, stratas_ys[i+1] - stratas_ys[i],
                               edgecolor=color_list[i % len(color_list)],
                               facecolor=color_list[i % len(color_list)],
                               alpha=0.7,
                               label=f"$A^{{{i+1}}}$"))
        # Rectangle for the stratum at x=1
        ax.add_patch(Rectangle((1 - rec_width/2, stratas_ys[i]),
                               rec_width, stratas_ys[i+1] - stratas_ys[i],
                               edgecolor=color_list[i % len(color_list)],
                               facecolor=color_list[i % len(color_list)],
                               alpha=0.7))
    # Scatter the stratified samples
    ax.scatter(np.full(max_iter, 0.5), Z05, c='green', marker='x', s=46,
               zorder=15, label='$Z_{2^{(k-1)}}$')
    ax.scatter(np.full(max_iter, 1.0), Z1, c='blue', marker='x', s=46,
               zorder=15, label='$Z_{2^k}$')
    ax.legend(loc="upper left")

    # Save the figure as a PDF
    filename = f"ch5_BrownianMotion_bisection_str_m_{nr_stratas}_str05_{strata05}_str1_{args.strata1}.pdf"
    fig.savefig(os.path.join(args.results_path, filename),bbox_inches='tight', pad_inches=0)
    ax.set_title("Strata Rectangles and Stratified Samples")
    # plt.close(fig)

    # ---------------------------------------------
    # Plot 2: Stratification with B(1/2) and B(1) markers
    # ---------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.05, 1.05)
    # Draw strata rectangles at x=1 (for full increment)
    for i in reversed(range(len(stratas_ys) - 1)):
        ax.add_patch(Rectangle((1 - rec_width/2, stratas_ys[i]),
                               rec_width, stratas_ys[i+1] - stratas_ys[i],
                               edgecolor=color_list[i % len(color_list)],
                               facecolor=color_list[i % len(color_list)],
                               alpha=0.7,
                               label=f"$A^{{{i+1}}}$",
                               zorder=10+i))
    # Compute the "combined" stratum for B(1/2)
    stratas_ys_B1_min = (stratas_ys[strata1] + stratas_ys[strata05]) / 2
    stratas_ys_B1_max = (stratas_ys[strata1+1] + stratas_ys[strata05+1]) / 2
    ax.add_patch(Rectangle((0.5 - rec_width/2, stratas_ys_B1_min),
                           rec_width, stratas_ys_B1_max - stratas_ys_B1_min,
                           edgecolor="#bd7676", facecolor="#bd7676",
                           alpha=0.7,
                           label=f"$C^{{{strata05+1},{strata1+1}}}$",
                           zorder=0))
    # Scatter B(1/2) and B(1) points: B(1/2) is taken as the average of Z05 and Z1.
    B05 = (Z05 + Z1) / 2
    ax.scatter(np.full(max_iter, 0.5), B05, c='red', marker='x', s=46,
               zorder=35, label='$B(1/2)$')
    ax.scatter(np.full(max_iter, 1.0), Z1, c='blue', marker='x', s=46,
               zorder=35, label="$B(1)$")
    ax.legend(loc="upper left")


    #plt.close(fig)

    # ---------------------------------------------
    # Plot 3: Full Brownian Motion Paths via Bisection
    # ---------------------------------------------
    #fig, ax = plt.subplots()
    # For each replicate, simulate a Brownian motion path using bisection.
    for iter in range(max_iter):
        B = np.zeros(2**k + 1)
        # Generate independent uniforms and compute corresponding normals
        U_all = rng.random(2**k + 1)
        Z = norm.ppf(U_all)
        # Replace specific increments with stratified values
        Z[2**(k-1)] = Z05[iter]
        Z[2**k] = Z1[iter]
        B[0] = 0
        B[2**k] = Z[2**k]
        # Construct the Brownian path by the bisection method
        for i in range(1, k+1):
            for j in range(2**(i-1)):
                idx = 2**(k-i) + j * 2**(k-i+1)
                B[idx] = (B[j * 2**(k-i+1)] + B[(j+1) * 2**(k-i+1)])/2 + 2**(-(i+1)/2)*Z[idx]
        t = np.linspace(0, 1, 2**k + 1)
        # Use a grayscale color that varies with the iteration index.
        ax.plot(t, B, color=(0.5 * iter / max_iter,
                              0.5 * iter / max_iter,
                              0.5 * iter / max_iter))

    # Save the figure as a PDF
    filename = f"ch5_BrownianMotion_bisection_str_m_{nr_stratas}_realizations_str05_{strata05}_str1_{args.strata1}.pdf"
    fig.savefig(os.path.join(args.results_path, filename),bbox_inches='tight', pad_inches=0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Brownian Motion")
    ax.set_title("Brownian Motion Paths via Stratified Bisection")
    plt.show()


if __name__ == "__main__":
    main()
