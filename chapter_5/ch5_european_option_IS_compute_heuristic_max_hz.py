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
## File: ch5_european_option_IS_compute_heuristic_max_hz.py         ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
This script uses the Newton-Raphson method to locate the maximum of the function
h(z) = (S0 * exp(mu_star + sigma*z) - K)_+ * phi(z) for a lognormally distributed asset.
"""

import numpy as np
from scipy.stats import norm
import argparse


def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Find the maximum of h(z) using the Newton-Raphson method."
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
                        help="Initial guess for z (default: %(default)s)")
    parser.add_argument('--tol', type=float, default=1e-6,
                        help="Tolerance for convergence (default: %(default)s)")
    parser.add_argument('--max_iter', type=int, default=100,
                        help="Maximum number of iterations (default: %(default)s)")
    # The last two entries: seed and results_path (if needed for future extensions)
    parser.add_argument('--seed', type=str, default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', type=str, default="results",
                        help="Folder for saving outputs (default: %(default)s)")
    return parser.parse_args()


def h(z, S0, K, mu_star, sigma):
    """
    h(z) = (S0 * exp(mu_star + sigma*z) - K)_+ * phi(z)
    where phi(z) is the standard normal PDF.
    """
    payoff = np.maximum(S0 * np.exp(mu_star + sigma * z) - K, 0)
    return payoff * norm.pdf(z)


def h_prime(z, S0, K, mu_star, sigma):
    """
    Derivative of h(z) with respect to z.
    """
    exp_term = S0 * np.exp(mu_star + sigma * z)
    payoff_derivative = sigma * exp_term - z * (exp_term - K)
    return payoff_derivative * norm.pdf(z)


def h_double_prime(z, S0, K, mu_star, sigma):
    """
    Second derivative of h(z) with respect to z.
    """
    exp_term = S0 * np.exp(mu_star + sigma * z)
    term1 = sigma ** 2 * exp_term - sigma * z * (exp_term - K)
    term2 = -(exp_term - K)
    return norm.pdf(z) * (term1 + term2) - z * h_prime(z, S0, K, mu_star, sigma)


def newton_raphson(f, f_prime, f_double, x0, tol, max_iter, S0, K, mu_star, sigma):
    """
    Newton-Raphson method for finding the critical point of h(z).
    """
    x = x0
    for _ in range(max_iter):
        f_val = f_prime(x, S0, K, mu_star, sigma)
        f_double_val = f_double(x, S0, K, mu_star, sigma)
        if abs(f_double_val) < 1e-12:  # Avoid division by zero
            print("Second derivative is too small. Stopping iteration.")
            break
        x_new = x - f_val / f_double_val
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    print("Newton-Raphson did not converge within the maximum number of iterations.")
    return x


def main():
    args = ParseArguments()

    # Calculate mu_star
    mu_star = args.r - (args.sigma ** 2) / 2

    # Print a brief introduction.
    print("This script applies the Newton-Raphson method to find the maximum of h(z),")
    print("which represents a payoff function weighted by the standard normal density.\n")

    # Optionally set the seed (not used here as no random sampling is involved)
    if args.seed.lower() != "none":
        np.random.seed(int(args.seed))

    # Compute the maximum point using Newton-Raphson
    z_max = newton_raphson(h, h_prime, h_double_prime, args.z_initial,
                           args.tol, args.max_iter, args.S0, args.K, mu_star, args.sigma)

    # Evaluate h(z) at the found maximum.
    h_max = h(z_max, args.S0, args.K, mu_star, args.sigma)

    print(f"Maximum z: {z_max:.6f}")
    print(f"Maximum value of h(z): {h_max:.6f}")


if __name__ == "__main__":
    main()
