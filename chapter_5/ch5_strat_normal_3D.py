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
## File: ch5_strat_normal_3D.py                                     ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Stratified Simulation for a Multivariate Standard Normal Distribution

We partition the simulation into m strata based on the chi-squared quantiles of the norm.
Each replication generates an n-dimensional standard normal vector which is normalized and then scaled
according to the quantile for its stratum. The resulting vectors are visualized in 3D.

Usage example:
    python stratified_multivariate.py --R 2000 --m 5 --n 3 --seed 31415 --results_path results
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from scipy.stats import chi2
from numpy.random import default_rng, PCG64

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Stratified simulation for multivariate standard normal (n-dimensions)."
    )
    parser.add_argument('--R', default="5000", help="Total number of replications (default: %(default)s)")
    parser.add_argument('--m', default="5", help="Number of strata (default: %(default)s)")
    parser.add_argument('--d', default="3", help="Dimension (default: %(default)s)")
    parser.add_argument('--seed', default="31415", help="Seed for RNG (use 'None' for no fixed seed, default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Directory for saving plots (default: %(default)s)")
    args = parser.parse_args()
    return args

def simulate_multivariate_strata(m, d, vecR, rng):
    """
    For each stratum i=0,...,m-1, for vecR[i] replications:
      - Generate an n-dimensional standard normal vector and normalize it.
      - Draw U ~ Uniform(0,1), set V = i/m + (1/m)*U.
      - Compute Rr = sqrt(chi2.ppf(V, df=n)) and set Z = normalized vector * Rr.
    Returns:
      Z: concatenated simulated vectors (shape: (sum(vecR), n))
      Y_i_strata: corresponding stratum indices.
    """
    Z = np.empty((0, d))
    Y_i_strata = np.array([])
    for i in range(m):
        ni = vecR[i]
        Y_i_strata = np.concatenate((Y_i_strata, i * np.ones(ni)))
        for _ in range(ni):
            vec = rng.normal(0, 1, d)
            norm_vec = np.linalg.norm(vec)
            Z_normed = vec / norm_vec if norm_vec != 0 else vec
            U = rng.random()
            V = i/m + (1/m) * U
            Rr = np.sqrt(chi2.ppf(V, df=d))
            Z_row = Z_normed * Rr
            Z = np.vstack((Z, Z_row))
    return Z, Y_i_strata.astype(int)

def main():
    args = ParseArguments()
    R = int(args.R)
    m = int(args.m)
    d = int(args.d)

    # Handle seed: if user inputs "None" then no fixed seed.
    seed = None if args.seed.strip().lower() == "none" else int(args.seed)
    # Prepare the main RNG (if a fixed seed is provided, we use it as base)
    rng = default_rng(PCG64(seed=seed))

    # Ensure results_path directory exists.
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Uniform allocation among strata.
    weights_uniform = np.ones(m) / m
    vecR = (weights_uniform * R).astype(int)

    # Simulate stratified multivariate samples.
    Z, Y_i_strata = simulate_multivariate_strata(m, d, vecR, rng)

    # Prepare a transformation matrix (Cholesky-like) for visualization.
    A_chol = (np.tril(np.ones((d, d))) / np.sqrt(d)).T
    Z_trans = np.dot(Z, A_chol)

    if d == 3:
        # Figure 1: Original Z, saved as PDF.
        fig1 = plt.figure(figsize=(9,9))
        ax1 = fig1.add_subplot(111, projection='3d')
        xs, ys, zs = Z[:, 0], Z[:, 1], Z[:, 2]
        for i in range(m):
            mask = (Y_i_strata == i)
            ax1.scatter(xs[mask], ys[mask], zs[mask], s=1, label=f"Str {i}")
        fig1.savefig(os.path.join(results_path, "ch5_strat_normal_3D_standard.pdf"), bbox_inches="tight", pad_inches=0)
        print("Saving ", results_path+"/ch5_strat_normal_3D_standard.pdf")
        ax1.set_title("Stratified Simulation (Original)")
        ax1.legend()

        # Figure 2: Transformed Z, saved as PDF.
        fig2 = plt.figure(figsize=(9,9))
        ax2 = fig2.add_subplot(111, projection='3d')
        xs2, ys2, zs2 = Z_trans[:, 0], Z_trans[:, 1], Z_trans[:, 2]
        for i in range(m):
            mask = (Y_i_strata == i)
            ax2.scatter(xs2[mask], ys2[mask], zs2[mask], s=1, label=f"Str {i}")
        fig2.savefig(os.path.join(results_path, "ch5_strat_normal_3D_Sigma.pdf"), bbox_inches="tight", pad_inches=0)
        print("Saving ", results_path+"/ch5_strat_normal_3D_Sigma.pdf")
        ax2.set_title("Stratified Simulation (Transformed)")
        ax2.legend()
    else:
        print("3D plots are available only for d=3.")

    plt.show()

if __name__ == "__main__":
    main()
