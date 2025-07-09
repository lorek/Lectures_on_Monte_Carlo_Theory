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
## File: ch5_asian_option_antithetic_control_variates.py            ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Pricing Asian Call Options using Antithetic and Control Variates Methods

This script computes option prices for different strike prices using two Monte Carlo methods:
1. Antithetic Variates
2. Control Variates

The parameters (including S0, r, sigma, n, R, Rpilot, and strike_prices as a comma separated list)
are set via command-line arguments. The random number generator uses PCG64 and a user-specified seed.
The results are displayed in a table using tabulate.
"""

import numpy as np
from numpy.random import PCG64, default_rng
from scipy.stats import norm
import argparse
import os
from tabulate import tabulate


def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Asian Call Option Pricing using Monte Carlo Methods"
    )
    parser.add_argument('--S0', type=float, default=100,
                        help="Initial asset price (default: %(default)s)")
    parser.add_argument('--r', type=float, default=0.05,
                        help="Risk-free interest rate (default: %(default)s)")
    parser.add_argument('--sigma', type=float, default=0.25,
                        help="Volatility (default: %(default)s)")
    parser.add_argument('--n', type=int, default=10,
                        help="Number of time steps (default: %(default)s)")
    parser.add_argument('--R', type=int, default=10000,
                        help="Number of main replications (default: %(default)s)")
    parser.add_argument('--Rpilot', type=int, default=1000,
                        help="Number of pilot replications (default: %(default)s)")
    parser.add_argument('--strike_prices', type=str, default="150,120,100,50,20",
                        help="Comma separated list of strike prices (default: %(default)s)")
    # Two last entries: seed and results_path
    parser.add_argument('--seed', type=str, default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', type=str, default="results",
                        help="Folder for saving outputs (default: %(default)s)")
    return parser.parse_args()


def black_scholes(muN, sigmaN2, K):
    """
    Calculate E[(G-K)_+] for Y = ln(X), X ~ N(muN, sigmaN2).
    """
    d1 = (-np.log(K) + muN + sigmaN2) / np.sqrt(sigmaN2)
    d2 = (-np.log(K) + muN) / np.sqrt(sigmaN2)
    return np.exp(muN + 0.5 * sigmaN2) * norm.cdf(d1) - K * norm.cdf(d2)


def asian_call_option_antithetic(S0, r, sigma, K, n, R, rng):
    """
    Antithetic variates estimation for an Asian call option.
    """
    dt = 1 / n  # Time step size
    mu_star = r - (sigma ** 2) / 2

    Z = rng.normal(size=(R // 2, n))
    Z_antithetic = -Z
    Z_combined = np.vstack((Z, Z_antithetic))

    # Simulate Brownian motion and asset paths
    B = np.cumsum(np.sqrt(dt) * Z_combined, axis=1)
    time_grid = np.linspace(1 / n, 1, n)
    S = S0 * np.exp(mu_star * time_grid + sigma * B)

    # Arithmetic average of asset prices
    A = np.mean(S, axis=1)
    payoffs = np.exp(-r) * np.maximum(A - K, 0)

    payoffs_non_antithetic = payoffs[:R // 2]
    payoffs_antithetic = payoffs[R // 2:]
    var_non_antithetic = np.var(payoffs_non_antithetic)
    var_antithetic = np.var(payoffs_antithetic)
    correlation = np.corrcoef(payoffs_non_antithetic, payoffs_antithetic)[0, 1]

    option_price = np.mean(payoffs)
    std_dev = np.std(payoffs) / np.sqrt(R)
    return option_price, 1.96 * std_dev


def asian_call_option_control_variates(S0, r, sigma, K, n, R, Rpilot, rng):
    """
    Control variates estimation for an Asian call option.
    """
    mu = r - sigma ** 2 / 2
    Y_j_pilot = np.zeros(Rpilot)
    YG_j_pilot = np.zeros(Rpilot)

    # Pilot simulation to estimate optimal coefficient
    for j in range(Rpilot):
        Z = rng.normal(0, 1, n)
        B = (1 / np.sqrt(n)) * np.cumsum(Z)
        S = S0 * np.exp(mu + sigma * B)
        A = np.mean(S)
        G = np.power(np.prod(S), 1 / n)
        Y_j_pilot[j] = np.exp(-r) * max(0, A - K)
        YG_j_pilot[j] = np.exp(-r) * max(0, G - K)

    cov_G_CMC = np.cov(YG_j_pilot, Y_j_pilot)[0, 1]
    var_G = np.var(YG_j_pilot)
    c_optimal = -cov_G_CMC / var_G


    print("K = ", K, ", covariance = ", cov_G_CMC, ", optimal c = ", c_optimal)
    # Calculate exact expectation of G using Black-Scholes formula (for control variate)
    mu_N = np.log(S0) + (r - sigma ** 2 / 2) * (n + 1) / (2 * n)
    sigma2_N = sigma ** 2 / (6 * n ** 2) * (n + 1) * (2 * n + 1)
    EG_exact = np.exp(-r) * black_scholes(mu_N, sigma2_N, K)

    Y_j = np.zeros(R)
    YG_j = np.zeros(R)
    Y_CV = np.zeros(R)
    for j in range(R):
        Z = rng.normal(0, 1, n)
        B = (1 / np.sqrt(n)) * np.cumsum(Z)
        S = S0 * np.exp(mu + sigma * B)
        A = np.mean(S)
        G = np.power(np.prod(S), 1 / n)
        Y_j[j] = np.exp(-r) * max(0, A - K)
        YG_j[j] = np.exp(-r) * max(0, G - K)
        Y_CV[j] = Y_j[j] + c_optimal * (YG_j[j] - EG_exact)

    mu_CMC = np.mean(Y_j)
    var_CMC = np.var(Y_j)
    mu_CV = np.mean(Y_CV)
    var_CV = np.var(Y_CV)
    return mu_CMC, 1.96 * np.sqrt(var_CMC), mu_CV, 1.96 * np.sqrt(var_CV)


def main():
    args = ParseArguments()

    # Create results folder if needed.
    os.makedirs(args.results_path, exist_ok=True)

    # Parse strike prices from comma-separated string.
    strike_prices = [float(s.strip()) for s in args.strike_prices.split(',')]

    # Set up the random number generator using PCG64.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed))

    results = []
    headers = ["K", "CMC Price", "CMC Error", "CV Price", "CV Error", "Anti Price", "Anti Error"]

    # For each strike price, compute the option prices.
    for K in strike_prices:
        mu_CMC, CMC_error, mu_CV, CV_error = asian_call_option_control_variates(
            args.S0, args.r, args.sigma, K, args.n, args.R, args.Rpilot, rng
        )
        anti_price, anti_error = asian_call_option_antithetic(
            args.S0, args.r, args.sigma, K, args.n, args.R, rng
        )
        results.append([K, mu_CMC, CMC_error, mu_CV, CV_error, anti_price, anti_error])

    # Display results in a formatted table using tabulate.
    print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="grid"))


if __name__ == "__main__":
    main()
