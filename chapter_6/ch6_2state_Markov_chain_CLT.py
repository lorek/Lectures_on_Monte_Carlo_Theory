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
## File: ch6_2state_Markov_chain_CLT.py                             ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Markov Chain Simulation and CLT Plots

For each (alpha, beta) pair provided via the command-line,
the script simulates a two-state Markov chain and produces two plots
(for two step intervals). The figures are saved in the folder specified by
--results_path.

Now uses np.random.default_rng(seed) with PCG for random number generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import argparse
import os
import time

def ParseArguments():
    parser = argparse.ArgumentParser(description="Markov chain simulation for CLT plots")
    parser.add_argument('--R', default="10000", help='Total number of points (default: %(default)s)')
    parser.add_argument('--R0', default="1", help='Starting step number (default: %(default)s)')
    parser.add_argument('--R1', default="250", help='First interval end step (default: %(default)s)')
    parser.add_argument('--R2', default="1000", help='Second interval start step (default: %(default)s)')
    parser.add_argument('--alphas', default="0.25,0.75",
                        help="Comma-separated list of alpha values (default: %(default)s)")
    parser.add_argument('--betas', default="0.125,0.875",
                        help="Comma-separated list of beta values (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving figures (default: %(default)s)")
    return parser.parse_args()

def simulate_markov(R, alpha, beta, rng):
    """
    Simulates a two-state Markov chain.
      - Generates an independent simulation Y from a uniform sample U with a threshold based on beta/(alpha+beta)
      - Simulates the Markov chain X with the specified transition probabilities.
    Returns:
      x_axis: steps from 1 to R
      Yhat, error: cumulative mean and confidence interval for Y
      Xhat, X_error: cumulative mean and confidence interval for X
    """
    U = rng.random(R)
    init_state = 1

    # For the independent simulation:
    prob0 = beta / (alpha + beta)
    Y = cp.copy(U)
    Y[0] = init_state
    Y[Y < prob0] = 0
    Y[Y >= prob0] = 1
    Yhat = np.cumsum(Y) / np.arange(1, R + 1)
    VarIID = alpha * beta / ((alpha + beta) ** 2)
    error = 1.96 * np.sqrt(VarIID) / np.sqrt(np.arange(1, R + 1))

    # For the Markov chain simulation:
    X = np.zeros(R)
    X[0] = init_state
    for t in range(R - 1):
        if X[t] == 0:
            # transition from 0: remains 0 with probability (1-alpha)
            X[t + 1] = 0 if U[t + 1] < (1 - alpha) else 1
        else:  # X[t] == 1
            # transition from 1: goes to 0 with probability beta
            X[t + 1] = 0 if U[t + 1] < beta else 1

    Xhat = np.cumsum(X) / np.arange(1, R + 1)
    AsVar_X = alpha * beta * (2 - alpha - beta) / ((alpha + beta) ** 3)
    X_error = 1.96 * np.sqrt(AsVar_X) / np.sqrt(np.arange(1, R + 1))

    x_axis = np.arange(1, R + 1)
    return x_axis, Yhat, error, Xhat, X_error

def main():
    args = ParseArguments()
    R = int(args.R)
    R0 = int(args.R0)
    R1 = int(args.R1)
    R2 = int(args.R2)

    # Parse the list of alphas and betas.
    alphas = [float(x) for x in args.alphas.split(',')]
    betas = [float(x) for x in args.betas.split(',')]
    if len(alphas) != len(betas):
        raise ValueError("The number of alphas must equal the number of betas.")

    # Set up results directory.
    os.makedirs(args.results_path, exist_ok=True)

    # Handle the seed.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(seed)

    # For each (alpha, beta) pair run the simulation and make plots.
    for idx, (alpha, beta) in enumerate(zip(alphas, betas)):
        # Run simulation.
        x_axis, Yhat, error, Xhat, X_error = simulate_markov(R, alpha, beta, rng)

        # Display the final (after R simulations) estimates and their 95% confidence intervals.
        final_ind_est = Yhat[-1]
        final_ind_ci = error[-1]
        final_markov_est = Xhat[-1]
        final_markov_ci = X_error[-1]
        print(f"\nFor alpha = {alpha}, beta = {beta}:")
        print(f"  Independent simulation estimate: {final_ind_est:.5f} ± {final_ind_ci:.5f}")
        print(f"  Markov chain simulation estimate:  {final_markov_est:.5f} ± {final_markov_ci:.5f}\n")

        # Define intervals for plots.
        intervals = [(R0, R1), (R2, R)]
        for j, (start, end) in enumerate(intervals):
            # Adjust indices (Python 0-indexing).
            s_idx = start - 1
            e_idx = end
            x_seg = x_axis[s_idx:e_idx]
            Yhat_seg = Yhat[s_idx:e_idx]
            error_seg = error[s_idx:e_idx]
            Xhat_seg = Xhat[s_idx:e_idx]
            X_error_seg = X_error[s_idx:e_idx]

            # Create plot.
            fig, ax = plt.subplots()
            # Plot independent simulation with confidence band.
            ax.plot(x_seg, Yhat_seg, label=r'$\hat{Y}_R^{\rm ind}$', color='b', alpha=1.0)
            ax.fill_between(x_seg, Yhat_seg - error_seg, Yhat_seg + error_seg,
                            color='b', alpha=0.25, label=r'conf. inter. for $\hat{Y}_R^{\rm ind}$')

            # Plot theoretical probability line.
            th_line = alpha / (alpha + beta)
            ax.plot(x_seg, np.full_like(x_seg, th_line, dtype=float),
                    label=r'$\alpha/(\alpha+\beta)$', color='r', alpha=0.5)

            # Plot Markov chain simulation with confidence band.
            ax.plot(x_seg, Xhat_seg, label=r'$\hat{Y}_R$', color='g', alpha=1.0)
            ax.fill_between(x_seg, Xhat_seg - X_error_seg, Xhat_seg + X_error_seg,
                            color='g', alpha=0.25, label=r'conf. inter. for $\hat{Y}_R$')



            # Save figure in results_path.

            fig_name = os.path.join(args.results_path, f"ch6_2state_Markov_chain_pair{idx}_plot{j}.pdf")
            plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
            ax.set_title(f"Alpha = {alpha}, Beta = {beta} | Steps {start} to {end}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Estimate")
            ax.legend()
            print(f"Saved plot: {fig_name}")



    # Optionally, show the last figure (or remove if running in batch mode)
    plt.show()

if __name__ == "__main__":
    main()
