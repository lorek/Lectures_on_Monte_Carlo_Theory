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
## File: ch5_european_option_IS_CE.py                               ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
This script performs two optimizations for selecting an optimal shift in importance sampling
for option pricing. It computes:
  - x_heuristic using Newton-Raphson on a weighted payoff function, and
  - x_star using a cross-entropy method.
Then, it compares a crude Monte Carlo estimate with importance sampling estimates using both shifts.
"""

import numpy as np
from numpy.random import PCG64, default_rng
from scipy.stats import norm
import argparse
import os
from tabulate import tabulate


def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Combined optimization and importance sampling for option pricing."
    )
    parser.add_argument('--S0', type=float, default=100,
                        help="Initial asset price (default: %(default)s)")
    parser.add_argument('--K', type=float, default=100,
                        help="Strike price (default: %(default)s)")
    parser.add_argument('--r', type=float, default=0.05,
                        help="Risk-free interest rate (default: %(default)s)")
    parser.add_argument('--sigma', type=float, default=0.25,
                        help="Volatility (default: %(default)s)")
    parser.add_argument('--z_initial', type=float, default=1.0,
                        help="Initial guess for Newton-Raphson (default: %(default)s)")
    parser.add_argument('--tol', type=float, default=1e-6,
                        help="Tolerance for convergence in Newton-Raphson (default: %(default)s)")
    parser.add_argument('--max_iter', type=int, default=100,
                        help="Maximum iterations for Newton-Raphson (default: %(default)s)")
    parser.add_argument('--M', type=int, default=1000,
                        help="Number of samples for cross-entropy optimization (default: %(default)s)")
    parser.add_argument('--R', type=int, default=10000,
                        help="Number of replications for Monte Carlo and IS (default: %(default)s)")
    # Last two entries:
    parser.add_argument('--seed', type=str, default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', type=str, default="results",
                        help="Folder for saving outputs (default: %(default)s)")
    return parser.parse_args()


# ---------------- Newton-Raphson for x_heuristic -----------------

def h(z, S0, K, mu_star, sigma):
    """
    h(z) = (S0 * exp(mu_star + sigma*z) - K)_+ * phi(z),
    where phi(z) is the standard normal PDF.
    """
    payoff = np.maximum(S0 * np.exp(mu_star + sigma * z) - K, 0)
    return payoff * norm.pdf(z)


def h_prime(z, S0, K, mu_star, sigma):
    """
    First derivative of h(z) with respect to z.
    """
    exp_term = S0 * np.exp(mu_star + sigma * z)
    payoff_derivative = sigma * exp_term - z * (exp_term - K)
    return payoff_derivative * norm.pdf(z)


def h_double_prime(z, S0, K, mu_star, sigma):
    """
    Second derivative of h(z) with respect to z.
    """
    exp_term = S0 * np.exp(mu_star + sigma * z)
    term1 = sigma**2 * exp_term - sigma * z * (exp_term - K)
    term2 = -(exp_term - K)
    return norm.pdf(z) * (term1 + term2) - z * h_prime(z, S0, K, mu_star, sigma)


def newton_raphson(f, f_prime, f_double, x0, tol, max_iter, S0, K, mu_star, sigma):
    """
    Newton-Raphson method to find a critical point for h(z).
    """
    x = x0
    for _ in range(max_iter):
        f_val = f_prime(x, S0, K, mu_star, sigma)
        f_double_val = f_double(x, S0, K, mu_star, sigma)
        if abs(f_double_val) < 1e-12:
            print("Second derivative is too small. Stopping iteration.")
            break
        x_new = x - f_val / f_double_val
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    print("Newton-Raphson did not converge within the maximum iterations.")
    return x


# ---------------- Cross-Entropy Optimization for x_star -----------------

def payoff(z, S0, K, mu_star, sigma):
    """
    Payoff function: (S0 * exp(mu_star + sigma*z) - K)_+.
    """
    return np.maximum(S0 * np.exp(mu_star + sigma * z) - K, 0)


def compute_x_star(Z, payoffs):
    """
    Compute x_star from samples: numerator/denominator.
    """
    numerator = np.sum(payoffs * Z)
    denominator = np.sum(payoffs)
    return numerator / denominator if denominator != 0 else 0.0


def cross_entropy_optimization(S0, K, r, sigma, M, rng):
    """
    Compute x_star using cross-entropy optimization.
    """
    mu_star = r - sigma**2 / 2
    Z = rng.normal(0, 1, M)
    payoffs_val = payoff(Z, S0, K, mu_star, sigma)
    return compute_x_star(Z, payoffs_val)


# ---------------- Importance Sampling and Crude Monte Carlo -----------------

def importance_sampling(S0, K, r, sigma, x_shift, R, rng):
    """
    Perform importance sampling with proposal N(x_shift, 1).
    """
    mu_star = r - sigma**2 / 2
    Z_is = rng.normal(x_shift, 1, R)
    likelihood_ratios = norm.pdf(Z_is, 0, 1) / norm.pdf(Z_is, loc=x_shift, scale=1)
    payoffs_is = payoff(Z_is, S0, K, mu_star, sigma) * likelihood_ratios
    I_is = np.exp(-r) * np.mean(payoffs_is)
    var_is = np.var(payoffs_is) / R
    return I_is, var_is


def crude_monte_carlo(S0, K, r, sigma, R, rng):
    """
    Perform crude Monte Carlo simulation.
    """
    mu_star = r - sigma**2 / 2
    Z_cmc = rng.normal(0, 1, R)
    payoffs_cmc = payoff(Z_cmc, S0, K, mu_star, sigma)
    I_cmc = np.exp(-r) * np.mean(payoffs_cmc)
    var_cmc = np.var(payoffs_cmc) / R
    return I_cmc, var_cmc


# ---------------- Main Function -----------------

def main():
    args = ParseArguments()
    os.makedirs(args.results_path, exist_ok=True)
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed))

    # Extract parameters into local variables
    S0 = args.S0
    K = args.K
    r = args.r
    sigma = args.sigma
    z_initial = args.z_initial
    tol = args.tol
    max_iter = args.max_iter
    M = args.M
    R = args.R

    # Calculate drift parameter for log-price
    mu_star = r - sigma**2 / 2

    # Print a brief description of what the code is doing.
    print("This script computes two optimal shifts for importance sampling in option pricing.\n"
          "It first computes x_heuristic via Newton-Raphson and x_star via cross-entropy, "
          "then compares crude Monte Carlo with importance sampling estimates using these shifts.\n")

    # Compute x_heuristic using Newton-Raphson on h(z)
    x_heuristic = newton_raphson(
        f=h, f_prime=h_prime, f_double=h_double_prime,
        x0=z_initial, tol=tol, max_iter=max_iter,
        S0=S0, K=K, mu_star=mu_star, sigma=sigma
    )

    # Compute x_star using cross-entropy optimization
    x_star = cross_entropy_optimization(S0, K, r, sigma, M, rng)

    # Display the computed optimal shifts.
    print(f"Computed x_heuristic (from Newton-Raphson): {x_heuristic:.6f}")
    print(f"Computed x_star (from cross-entropy):         {x_star:.6f}\n")

    # Compute Crude Monte Carlo Estimate
    I_cmc, var_cmc = crude_monte_carlo(S0, K, r, sigma, R, rng)

    # Perform Importance Sampling with both x_star and x_heuristic
    I_is_x_star, var_is_x_star = importance_sampling(S0, K, r, sigma, x_star, R, rng)
    I_is_x_heuristic, var_is_x_heuristic = importance_sampling(S0, K, r, sigma, x_heuristic, R, rng)


    # Prepare updated table for display.
    table = [
        ["Estimate hat{Y}", f"{I_cmc:.4f}", f"{I_is_x_heuristic:.4f}", f"{I_is_x_star:.4f}"],
        ["b=1.96sqrt{hat{S}}$", f"{1.96 * np.sqrt(var_cmc):.4f}", f"{1.96 * np.sqrt(var_is_x_heuristic):.4f}",
         f"{1.96 * np.sqrt(var_is_x_star):.4f}"],
        ["Var(Y^{CMC})/Var(Y^{IS})", "--", f"{(var_cmc / var_is_x_heuristic):.4f}",
         f"{(var_cmc / var_is_x_star):.4f}"],
    ]
    headers = ["", "CMC", "IS ($x_{heuristic}$)", "IS ($x_{CE}$)"]

    print("\n--- Updated Option Pricing Estimates ---")
    print(tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
