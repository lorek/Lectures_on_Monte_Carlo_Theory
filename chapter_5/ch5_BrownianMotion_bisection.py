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
## File: ch5_BrownianMotion_bisection.py                            ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Simulate Brownian motion via the bisection method.

This script:
  1. Simulates Brownian motion on a fine grid t = j/2^k for j = 0,...,2^k using
     the bisection method (default k=15).
  2. Extracts a coarse grid corresponding to indices j*2^(s) (with s=k-2 by default),
     which yields 5 points: B(0), B(1/4), B(2/4), B(3/4), B(1).
  3. Displays these coarse values in horizontal tables.
  4. Then, repeats (4 additional times) the simulation of the full fine grid,
     but forcing the coarse-grid values to remain the same as in the original simulation.
     It produces 5 plots (each in a different shade) of the Brownian motion.

Usage example:
    python brownian_bisection.py --k 15 --s 13 --seed 31415 --results_path results
    (To disable seeding, use --seed None)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import os
from tabulate import tabulate
from numpy.random import default_rng, PCG64


def ParseArguments():
    parser = argparse.ArgumentParser(description="Simulate Brownian motion via bisection.")
    parser.add_argument('--k', default="15", help="Bisection level k (default: %(default)s)")
    parser.add_argument('--s', default=None, help="Coarse grid exponent s (default: k-2)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Directory for saving plots (default: %(default)s)")
    args = parser.parse_args()
    return args

 


def simulate_brownian_bisection(k, rng):
    """
    Simulate Brownian motion on grid indices 0,...,2^k (i.e. at times j/2^k).
    Uses the bisection method (Algorithm 11). Returns the full array B and the full innovation array Z.
    """
    N = 2 ** k + 1
    B = np.zeros(N)
    U = rng.random(N)
    Z = norm.ppf(U)
    # Set B at the final index; and force Z[0] = 0.
    B[-1] = Z[-1]
    Z[0] = 0
    # Loop over bisection levels i = 1,...,k.
    for i in range(1, k + 1):
        for j in range(2 ** (i - 1)):
            idx = 2 ** (k - i) + j * 2 ** (k - i + 1)
            left = B[j * 2 ** (k - i + 1)]
            right = B[(j + 1) * 2 ** (k - i + 1)]
            B[idx] = (left + right) / 2 + 2 ** (-(i + 1) / 2) * Z[idx]
    return B, Z


def extract_coarse_grid(B, Z, k, s):
    """
    Extract coarse-grid values from full simulation.
    The coarse grid is given by indices j*2^s for j=0,...,2^(k-s).
    Returns:
      indices, B_coarse, and Z_coarse.
    """
    num_points = 2 ** (k - s) + 1
    indices = np.array([j * 2 ** s for j in range(num_points)])
    B_coarse = B[indices]
    Z_coarse = Z[indices]
    return indices, B_coarse, Z_coarse


def main():
    args = ParseArguments()
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.strip().lower() == "none" else int(args.seed)
    rng =  default_rng(PCG64(seed=seed))

     
    k = int(args.k)
    s = int(args.s) if args.s is not None else k - 2  # default: s = k - 2

    # Simulate full Brownian motion using bisection.
    B_full, Z_full = simulate_brownian_bisection(k, rng)
    indices_coarse, B_coarse, Z_coarse = extract_coarse_grid(B_full, Z_full, k, s)

    # Create horizontal table for coarse-grid innovations (Z_coarse), excluding index 0 if desired.
    headers_Z = [f"Z at t={idx}/2^{k}" for idx in indices_coarse[1:]]
    table_Z = tabulate([["{:.4f}".format(val) for val in Z_coarse[1:]]], headers=headers_Z, tablefmt="grid")
    print("\nCoarse-grid Innovations (Z):")
    print(table_Z)

    # Create horizontal table for coarse-grid B values.
    headers_B = [f"B({idx}/2^{k})" for idx in indices_coarse]
    table_B = tabulate([["{:.4f}".format(val) for val in B_coarse]], headers=headers_B, tablefmt="grid")
    print("\nCoarse-grid Brownian Motion Values (B):")
    print(table_B)



    # Now, produce the main plot: plot the full Brownian motion using B_full.
    x_full = np.arange(0, 2 ** k + 1)
    xvals = x_full/(2**k)

    fig_main, ax_main = plt.subplots(figsize=(10, 5))

    #ax_main.plot(x_full, B_full, color="C0", alpha=1, label="Full Brownian Motion")
    ax_main.plot(xvals, B_full, color="C0", alpha=1, label="Full Brownian Motion")

    #ax_main.scatter(x_full, B_full, color='C0', s=1)
    ax_main.scatter(xvals[indices_coarse], B_full[indices_coarse], color='red', s=40, zorder=5)

    # Adjust x-axis from 0 to 1
    ax_main.set_xlim([-0.02, 1.02])

    # Set the tick marks at the fractions you want
    ax_main.set_xticks([0, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 6 / 8, 7 / 8, 1])
    ax_main.set_xticklabels(["0", "1/8", "1/4", "3/8", "1/2", "5/8", "6/8", "7/8", "1"])


    main_plot_file = os.path.join(results_path, "ch5_BrownianMotion_bisection_1realization.pdf")
    plt.savefig(main_plot_file, bbox_inches='tight', pad_inches=0)
    ax_main.set_xlabel("t")
    ax_main.set_ylabel("B(t)")
    ax_main.set_title(f"Simulated Brownian Motion (k={k}, {2 ** k + 1} points)")
    print(f"\nFull Brownian motion plot saved to: {main_plot_file}")



    # Now, produce 5 additional plots.
    # For each new plot, we simulate the fine grid using new innovations for indices beyond the coarse grid,
    # but force the coarse-grid values (at indices_coarse) to be the same as originally.
    num_plots = 5
    fig_files = []

    fig, ax = plt.subplots(figsize=(10,5))
    #previous plot
    #ax.plot(x_full, B_full, color="C0", alpha=1 )

    # Adjust x-axis from 0 to 1
    ax.set_xlim([-0.02, 1.02])

    # Set the tick marks at the fractions you want
    ax.set_xticks([0, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 6 / 8, 7 / 8, 1])
    ax.set_xticklabels(["0", "1/8", "1/4", "3/8", "1/2", "5/8", "6/8", "7/8", "1"])

    for iter in range(num_plots-1):
        print("Iteration ", iter," ...")
        # Generate new full innovations for the fine grid.
        U_new = rng.random(2 ** k + 1)
        Z_new = norm.ppf(U_new)
        # Create a new array B_new.
        B_new = np.zeros(2 ** k + 1)
        # Force the coarse-grid values to remain as before.
        B_new[indices_coarse] = B_coarse
        # Now, for the remaining indices, perform bisection calculations for levels i = (k-s+1) to k.
        for i in range(k - s + 1, k + 1):
            for j in range(2 ** (i - 1)):
                idx = 2 ** (k - i) + j * 2 ** (k - i + 1)
                # Only update indices that are not in the coarse grid.
                if idx not in indices_coarse:
                    left = B_new[j * 2 ** (k - i + 1)]
                    right = B_new[(j + 1) * 2 ** (k - i + 1)]
                    B_new[idx] = (left + right) / 2 + 2 ** (-(i + 1) / 2) * Z_new[idx]
        # Plot B_new.

        #ax.plot(x_full, B_new, color='C{}'.format(iter), alpha=0.6, label=f"Iteration {iter + 1}" )
        ax.plot(xvals, B_new, color=np.array([1/2*iter/5,1/2*iter/5,1/2*iter/5]), alpha=0.6, label=f"Iteration {iter + 1}")
        ax.scatter(xvals, B_new,   s=1, color=np.array([1/2*iter/5,1/2*iter/5,1/2*iter/5]))


    ax.scatter(xvals[indices_coarse], B_full[indices_coarse], color='red', s=40, zorder=5)

    main5_plot_file = os.path.join(results_path, "ch5_BrownianMotion_bisection_5realizations.pdf")
    plt.savefig(main5_plot_file, bbox_inches='tight', pad_inches=0)
    ax.set_xlabel("Grid Index (j)")
    ax.set_ylabel("B(j/2^k)")
    ax.set_title(f"Brownian Motion (Iteration {iter + 1}, k={k})")
    print(f"\nFull Brownian motion plot saved to: {main5_plot_file}")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
