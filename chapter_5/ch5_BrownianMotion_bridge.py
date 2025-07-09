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
## File: ch5_BrownianMotion_bridge.py                               ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Simulate Brownian bridges using the bisection method.

This script:
  1. Simulates Brownian bridges on a fine grid t = j/2^k for j = 0,...,2^k.
  2. Repeats the simulation `nr_simulations` times, plotting all realizations on a single figure.
  3. Each Brownian bridge satisfies B(0) = B(1) = 0.
  4. The resulting plot is saved as a PDF.

Usage:
    python ch5_BrownianMotion_bridge.py --k 15 --nr_simulations 10 --seed 12345 --results_path results
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import os
from numpy.random import default_rng, PCG64


def ParseArguments():
    parser = argparse.ArgumentParser(description="Simulate Brownian bridges via bisection.")
    parser.add_argument('--k', type=int, default=15, help="Level of grid resolution (default: %(default)s)")
    parser.add_argument('--nr_simulations', type=int, default=10, help="Number of bridge realizations (default: %(default)s)")
    parser.add_argument('--seed', type=str, default="31415", help="Seed for PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', type=str, default="results", help="Directory for saving output plots (default: %(default)s)")
    return parser.parse_args()


def simulate_brownian_bridge(k, rng):
    """
    Simulate a Brownian bridge on grid j/2^k, j=0,...,2^k using bisection.
    Enforces B(0)=0 and B(1)=0.
    """
    N = 2**k + 1
    B = np.zeros(N)
    U = rng.random(N)
    Z = norm.ppf(U)
    Z[0] = 0
    B[0] = 0
    B[-1] = 0  # Bridge endpoint

    for i in range(1, k + 1):
        for j in range(2 ** (i - 1)):
            idx = 2 ** (k - i) + j * 2 ** (k - i + 1)
            left = B[j * 2 ** (k - i + 1)]
            right = B[(j + 1) * 2 ** (k - i + 1)]
            B[idx] = (left + right) / 2 + 2 ** (-(i + 1) / 2) * Z[idx]
    return B


def main():
    args = ParseArguments()
    os.makedirs(args.results_path, exist_ok=True)

    seed = None if args.seed.strip().lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed=seed))

    k = args.k
    nr_simulations = args.nr_simulations
    N = 2 ** k + 1
    t_vals = np.linspace(0, 1, N)

    fig, ax = plt.subplots(figsize=(10, 5))

    for sim in range(nr_simulations):
        B = simulate_brownian_bridge(k, rng)
        color = np.array([0.5 * sim / nr_simulations] * 3)
        ax.plot(t_vals, B, color=color, alpha=0.6)


    ax.set_xlim([-0.02, 1.02])

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

      # Set the tick marks at the fractions you want
    ax.set_xticks([0, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 6 / 8, 7 / 8, 1])
    ax.set_xticklabels(["0", "1/8", "1/4", "3/8", "1/2", "5/8", "6/8", "7/8", "1"])

    output_file = os.path.join(args.results_path, "ch5_BrownianMotion_bridge.pdf")
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    ax.set_xlabel("t")
    ax.set_ylabel("B(t)")
    ax.set_title(f"{nr_simulations} Brownian bridge realizations (k={k})")
    print(f"Plot saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
