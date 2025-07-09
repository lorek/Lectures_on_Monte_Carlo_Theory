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
## File: ch6_hardcore_model_clt.py                                  ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
CLT Simulation for a Hardcore Model (Example 7.13 in hc_clt.pdf)

The simulation proceeds on an n×n grid. At each replication a random grid point
is selected and, if it has no neighbours (according to a four-neighbour criterion),
a particle is placed with probability λ/(λ+1). The cumulative mean of the total
number of particles is computed and its convergence monitored.

The script also computes approximate confidence intervals based on the lagged
autocovariances for different lag truncation parameters N, and saves various plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy as cp
from matplotlib.ticker import MultipleLocator
import argparse
import os
import time

def ParseArguments():
    parser = argparse.ArgumentParser(description="CLT Simulation for a Hardcore Model (Example 7.13)")
    parser.add_argument('--R', default="3000", help='Total number of replications (default: %(default)s)')
    parser.add_argument('--R0', default="1", help='Starting replication (default: %(default)s)')
    parser.add_argument('--lambd', default="1", help='Lambda parameter (default: %(default)s)')
    parser.add_argument('--n', default="8", help='Grid dimension n (default: %(default)s) for an n x n grid')
    parser.add_argument('--seed', default="31415", help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Folder for saving figures (default: %(default)s)")
    return parser.parse_args()

def compute_nr_neighbours(xi, i, j):
    n = xi.shape[0]
    nr = 0
    if i < n - 1:
        nr += xi[i+1, j]
    if i > 0:
        nr += xi[i-1, j]
    if j < n - 1:
        nr += xi[i, j+1]
    if j > 0:
        nr += xi[i, j-1]
    return nr

def lagged_auto_cov(Xi, t):
    """
    Compute empirical autocovariance with lag t for the series Xi.
    Defined as: (1/(N-t)) * sum_{i=0}^{N-t-1} (x_i - x̄)*(x_{i+t} - x̄)
    """
    N = len(Xi)
    X_bar = np.mean(Xi)
    # Pad series to align indices
    end_series = np.zeros(N+t)
    end_series[:N] = Xi - X_bar
    start_series = np.zeros(N+t)
    start_series[t:] = Xi - X_bar
    return np.sum(start_series * end_series) / (N - t)

def compute_mean_var(Y):
    R = Y.shape[0]
    Yhat = np.cumsum(Y) / np.arange(1, R+1)
    N_trunc = 100  # maximum lag to consider in the autocovariance sum
    varr = np.zeros(R)
    for r in range(1, R):
        Yhat_r = np.mean(Y[:r])
        # start with variance at lag 0
        cov_sum = np.var(Y[:r])
        for k in range(1, min(N_trunc, r)):
            cov_sum += 2 * np.mean((Y[:(r-k)] - Yhat_r) * (Y[k:r] - Yhat_r))
        varr[r] = cov_sum
    return Yhat, varr

def main():
    args = ParseArguments()
    R = int(args.R)
    R0 = int(args.R0)
    lambd = float(args.lambd)
    n = int(args.n)

    # Set up results directory.
    os.makedirs(args.results_path, exist_ok=True)

    # Set up PRNG using PCG (default_rng uses PCG64 by default).
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(seed)

    # Initialize grid.
    xi = np.zeros((n, n))

    # Pre-generate all random indices and coin tosses.
    Ui = (np.floor(n * rng.random(R))).astype(int)
    Uj = (np.floor(n * rng.random(R))).astype(int)
    Ucoin = rng.random(R)

    # Array to record total number of particles at each replication.
    Y1 = np.zeros(R)

    # For snapshots.
    xi_step2 = None
    xi_step20 = None
    xi_step_last = None

    for k in range(R):
        i = Ui[k]
        j = Uj[k]
        nr_neigh = compute_nr_neighbours(xi, i, j)
        if nr_neigh == 0:
            if Ucoin[k] < lambd / (lambd + 1):
                xi[i, j] = 1
        Y1[k] = np.sum(xi)
        if k == 2:
            xi_step2 = cp.copy(xi)
        if k == 20:
            xi_step20 = cp.copy(xi)
        if k == R - 1:
            xi_step_last = cp.copy(xi)

    Y1hat = np.cumsum(Y1) / np.arange(1, R+1)

    val_last_step = Y1[-1]

    # Save grid snapshots.
    grid_snapshots = [
        (xi_step2, "step2"),
        (xi_step20, "step20"),
        (xi_step_last, f"step{R}")
    ]
    for snap, label in grid_snapshots:
        fig, ax = plt.subplots(figsize=(5, 5))
        x_coords, y_coords = np.nonzero(snap)
        ax.scatter(x_coords+1, y_coords+1, s=201, alpha=1, c='b')
        ax.set_xlim([1, n])
        ax.set_ylim([1, n])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.grid(color="black", linewidth=1.25, alpha=0.3)
        fig_name = os.path.join(args.results_path, f"ch6_hardcore_{label}.pdf")
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        print(f"Saved grid snapshot: {fig_name}, \t\t number of particles {np.sum(snap)}")
        ax.set_title(f"{label}: number of particles = {np.sum(snap)}")



    # Compute confidence intervals based on lagged autocovariances.
    t0 = 500  # starting index for CI computation
    x_axis = np.arange(1, R+1)
    Y1hat_full, _ = compute_mean_var(Y1)
    # Prepare a plot of Y1hat and various CI boundaries for different lag truncation parameters N.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, Y1hat_full, label="Estimation", color='b', alpha=1)

    clrs = ['r','c','m','y','k']
    for cntr, N in enumerate([2, 10, 100, 200, 250]):
        print("Computations for N = ", N, " ... ")
        asvars = np.zeros(R - t0)
        for t in range(t0, R):
            YY = Y1[:t]
            asvar = lagged_auto_cov(YY, 0)
            for k in range(1, N+1):
                asvar += 2 * lagged_auto_cov(YY, k)
            asvars[t - t0] = asvar
        error = 1.96 * np.sqrt(asvars) / np.sqrt(np.arange(t0+1, R+1))
        ax.plot(x_axis[t0:], Y1hat_full[t0:] - error, alpha=1, c=clrs[cntr], label=f"N = {N}")
        ax.plot(x_axis[t0:], Y1hat_full[t0:] + error, alpha=1, c=clrs[cntr])
    ax.legend()
    fig_name = os.path.join(args.results_path, "ch6_hardcore_CI_multiple_N.pdf")
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print(f"Saved CI plot (multiple N): {fig_name}")

    # Plot with a single CI band (using the last N from the loop)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_axis, Y1hat_full, label="Estimation", color='b', alpha=1)
    # Recompute error for N = 250 (last value in our list)
    print("Recomputing for N = 250 ...")
    asvars = np.zeros(R - t0)
    for t in range(t0, R):
        YY = Y1[:t]
        asvar = lagged_auto_cov(YY, 0)
        for k in range(1, 250+1):
            asvar += 2 * lagged_auto_cov(YY, k)
        asvars[t - t0] = asvar
    error = 1.96 * np.sqrt(asvars) / np.sqrt(np.arange(t0+1, R+1))
    ax.fill_between(x_axis[t0:], Y1hat_full[t0:]-error, Y1hat_full[t0:]+error,
                    alpha=0.25, label="95% Confidence Interval", color=[0.8,0.3,0.2])
    ax.plot(x_axis[t0:], Y1hat_full[t0:]-error, alpha=1, c='k', label="N = 250")
    ax.plot(x_axis[t0:], Y1hat_full[t0:]+error, alpha=1, c='k')
    ax.legend()
    fig_name = os.path.join(args.results_path, "ch6_hardcore_CI_final.pdf")
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print(f"Saved final CI plot: {fig_name}")


    print(f"\nFinal estimated Y1hat_R = {Y1hat_full[-1]:.5f} with 95% CI: ({Y1hat_full[-1]-error[-1]:.5f}, {Y1hat_full[-1]+error[-1]:.5f})")
    print(f"\nValue at last step: {val_last_step}")
    print("\n")
    plt.show()

if __name__ == "__main__":
    main()
