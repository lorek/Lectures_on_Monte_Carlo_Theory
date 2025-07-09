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
## File: ch5_IS_coin_flipping.py                                    ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Importance Sampling for Rare Events in Coin Flipping

This script estimates the probability I(n) = P(Sₙ > n(μ+ε)) for coin flips using:
    - Crude Monte Carlo (CMC)
    - Importance Sampling (IS)

The exact probability is computed via the binomial CDF. For the IS estimator,
a biased coin with probability β = μ + ε is used, and each outcome is weighted
by the likelihood ratio to correct for the change in measure.

Example usage:
    python ch5_ImportanceSampling_coin.py --n 50 --epsilon 0.1,0.25 --mu 0.5 --R 1000 --seed 31415
"""

import numpy as np
import argparse
import os
from scipy.stats import binom
from numpy.random import default_rng, PCG64
from tabulate import tabulate
import math


def ParseArguments():
    parser = argparse.ArgumentParser(description="Importance Sampling for Coin Flipping")
    parser.add_argument('--n', default="50", help="Number of coin flips (default: %(default)s)")
    parser.add_argument('--epsilon', default="0.25,0.1",
                        help="Comma-separated list of shift parameters epsilon (default: %(default)s)")
    parser.add_argument('--mu', default="0.5", help="Mean of the coin (default: %(default)s)")
    parser.add_argument('--R', default="1000", help="Number of replications (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Folder for saving outputs (default: %(default)s)")
    return parser.parse_args()


def main():
    args = ParseArguments()
    # Handle seed input.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed=seed))

    # Ensure results directory exists.
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Parse simulation parameters.
    n = int(args.n)
    epsilon_list = list(map(float, args.epsilon.split(',')))
    mu = float(args.mu)
    R = int(args.R)

    # Table to store results for each epsilon.
    table = []

    headers = ["n", "Epsilon",  "Exact I(n)", "Estimator", "Mean", "Variance"]

    for epsilon in epsilon_list:
        beta = mu + epsilon
        # Determine threshold: floor(n*(mu+epsilon))
        threshold_value = n * (mu + epsilon)
        threshold = math.floor(threshold_value)

        # Compute exact probability using binomial distribution (p=0.5 for fair coin)
        I_exact = 1 - binom.cdf(threshold, n, 0.5)

        # -------------------------
        # Crude Monte Carlo (CMC)
        # -------------------------
        X_cmc = rng.binomial(1, 0.5, size=(R, n))
        S_cmc = X_cmc.sum(axis=1)
        Y_cmc = (S_cmc > threshold).astype(float)
        cmc_mean = np.mean(Y_cmc)
        cmc_var = np.var(Y_cmc, ddof=1)

        # -------------------------
        # Importance Sampling (IS)
        # -------------------------
        X_is = rng.binomial(1, beta, size=(R, n))
        S_is = X_is.sum(axis=1)
        weights = np.zeros(R)
        mask = (S_is > threshold)
        weights[mask] = ((0.5) ** n) / ((beta ** S_is[mask]) * ((1 - beta) ** (n - S_is[mask])))
        is_mean = np.mean(weights)
        is_var = np.var(weights, ddof=1)

        # Append the row for this epsilon.
        table.append([n, epsilon,  f"{I_exact:.10f}", "CMC", f"{cmc_mean:.10e}", f"{cmc_var:.10e}"])
        table.append(["", "",   "", "IS",f"{is_mean:.10e}", f"{is_var:.10e}"])

    # Display the results using Tabulate.
    print("\n--- Simulation Results ---")
    print(f"Parameters: n={n}, mu={mu}, R={R}")
    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
