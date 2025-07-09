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
## File: ch5_cross_entropy_normal_1D.py                             ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Cross-Entropy Method for Estimating P(X > γ_target) for X ~ N(0,1)

This script implements the cross-entropy (CE) method for rare event simulation (Example 1.50)
to estimate I = P(X > 4) using a family of proposal distributions N(θ,1). It performs the CE
iterations to obtain an optimal θ, then computes final estimates using three approaches:
  1. Crude Monte Carlo (CMC) with N(0,1),
  2. Importance Sampling (IS) using proposal N(γ_target, 1) with γ_target = 4 (default),
  3. IS using the optimal θ from the CE method.

The evolution of γ and θ during the CE procedure is plotted (Figure 18) and saved as PDF.
Results are displayed in a table using the tabulate package.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import os
from numpy.random import default_rng, PCG64
from tabulate import tabulate
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

def ParseArguments():
    parser = argparse.ArgumentParser(description="Cross-Entropy Method for Rare Event Simulation (Example 1.50)")
    parser.add_argument('--T', default="50", help="Maximum number of iterations (default: %(default)s)")
    parser.add_argument('--rho', default="0.55", help="Quantile level (default: %(default)s)")
    parser.add_argument('--R_prime', default="10000", help="Sample size for parameter updates (default: %(default)s)")
    parser.add_argument('--R', default="10000", help="Sample size for final estimation (default: %(default)s)")
    parser.add_argument('--gamma_target', default="4",
                        help="Target gamma value for stopping criteria (default: %(default)s)")
    parser.add_argument('--alpha', default="0.4", help="Smoothing parameter for update (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving output figures (default: %(default)s)")
    parser.add_argument('--seed', default="31415", help="Seed for the PRNG (default: %(default)s)")
    args = parser.parse_args()
    return args


# Density functions for N(0,1) and N(theta,1)
def f0(x):
    return norm.pdf(x, loc=0, scale=1)


def f_theta(x, theta):
    return norm.pdf(x, loc=theta, scale=1)


def cross_entropy_step(theta_prime, R_prime, rho, rng):
    """
    Perform one cross-entropy step:
      - Draw samples from N(theta_prime,1)
      - Compute the (1-rho)-quantile, gamma_hat, of the samples.
      - Compute importance weights: w = f0(x)/f_theta(x, theta_prime)
      - Update theta_prime as the weighted average of samples above gamma_hat.
    """
    samples = rng.normal(loc=theta_prime, scale=1, size=R_prime)
    samples_sorted = np.sort(samples)
    gamma_hat = samples_sorted[int((1 - rho) * R_prime)]
    weights = f0(samples) / f_theta(samples, theta_prime)

    valid_mask = samples >= gamma_hat
    valid_samples = samples[valid_mask]
    valid_weights = weights[valid_mask]

    updated_theta = np.average(valid_samples, weights=valid_weights) if valid_samples.size > 0 else theta_prime
    return updated_theta, gamma_hat


def objective(theta, gamma_target):
    """
    Compute the objective function value:
       F(theta) = int_{x=gamma_target}^{inf} (x - theta)^2 * exp(-x^2/2) dx.

    Parameters:
        theta: float
            The parameter at which to evaluate the function.
        gamma_target: float
            The lower limit of integration.

    Returns:
        float: The value of the integral.
    """
    integrand = lambda x: (x - theta) ** 2 * np.exp(-x ** 2 / 2)
    val, _ = quad(integrand, gamma_target, np.inf)
    return val


def find_optimal_theta(gamma_target, theta_bounds=None):
    """
    Find theta^* that minimizes the objective function:
       F(theta) = int_{gamma_target}^{inf} (x - theta)^2 * exp(-x^2/2) dx.

    Parameters:
        gamma_target: float
            The lower limit of integration (e.g., 4).
        theta_bounds: tuple (optional)
            A tuple (theta_min, theta_max) specifying bounds for theta.
            If not provided, defaults to (gamma_target, gamma_target + 10).

    Returns:
        float: The optimal theta value.
    """
    if theta_bounds is None:
        theta_bounds = (gamma_target, gamma_target + 10)

    res = minimize_scalar(objective, args=(gamma_target,), bounds=theta_bounds, method='bounded')
    return res.x


def main():
    args = ParseArguments()
    T = int(args.T)
    rho = float(args.rho)
    R_prime = int(args.R_prime)
    R = int(args.R)
    gamma_target = float(args.gamma_target)
    alpha = float(args.alpha)
    results_path = args.results_path
    seed = None if args.seed.lower() == "none" else int(args.seed)

    os.makedirs(results_path, exist_ok=True)
    rng = default_rng(PCG64(seed=seed))



    # Cross-Entropy initialization: start with theta0 = 0.
    theta0 = 0.0
    theta_prime = theta0
    theta_history = [theta_prime]
    gamma_history = []


    print("Starting Cross-Entropy iterations with T =", T)
    t_final = T  # default if no early stop

    for t in range(1, T + 1):
        updated_theta, gamma_hat = cross_entropy_step(theta_prime, R_prime, rho, rng)
        # Check if gamma_hat exceeds gamma_target.
        if gamma_hat > gamma_target:
            t_final = t - 1
            break
        theta_prime = alpha * updated_theta + (1 - alpha) * theta_prime
        theta_history.append(np.round(theta_prime,3))
        gamma_history.append(np.round(gamma_hat,3))

    print("Consecutive thetas:\n", [float(x) for x in theta_history])
    print("Consecutive gammas:\n", [float(x) for x in gamma_history])
    print(">>>>>>>> ", len(theta_history),len(gamma_history))
    print("Final iteration:", t_final)

    theta_ce = theta_prime

    final_gamma = gamma_history[-1] if gamma_history else None
    print("\nCE method completed.")
    print(f"Final theta (optimal) from CE: {theta_ce:.4f}")
    # Find optimal theta numerically:
    theta_star = find_optimal_theta(gamma_target)
    print(f"Optimal theta^* (numercial sol. of eq) for gamma_target = {gamma_target}: {theta_star:.4f}")

    print(f"Final gamma from CE: {final_gamma:.4f}")


    I_true = 1-norm.cdf(gamma_target)
    print("I_true = P(X>",gamma_target,") = ", I_true)

    # ESTIMATIONS:

    # 1. Crude Monte Carlo (CMC) using N(0,1)
    samples_cmc = rng.normal(loc=0, scale=1, size=R)
    cmc_indicator = (samples_cmc > 4).astype(float)
    cmc_estimate = np.mean(cmc_indicator)
    cmc_variance = np.var(cmc_indicator, ddof=1) / R
    cmc_abs_err = np.abs(cmc_estimate-I_true)

    # 2. Importance Sampling with proposal N(gamma_target, 1) i.e. theta = 4.
    samples_theta_target = rng.normal(loc=gamma_target, scale=1, size=R)
    weights_theta_target = f0(samples_theta_target) / f_theta(samples_theta_target, gamma_target)
    is_target_estimate = np.mean(weights_theta_target * (samples_theta_target > 4))
    is_target_variance = np.var(weights_theta_target * (samples_theta_target > 4), ddof=1) / R
    is_abs_err = np.abs(is_target_estimate - I_true)

    # 4. Importance Sampling with proposal N(theta_star, 1) -- optimal solution (obtained numerically)
    samples_theta_star = rng.normal(loc=theta_star, scale=1, size=R)
    weights_theta_star = f0(samples_theta_star) / f_theta(samples_theta_star, theta_star)
    is_star_estimate = np.mean(weights_theta_star * (samples_theta_star > 4))
    is_star_variance = np.var(weights_theta_star * (samples_theta_star > 4), ddof=1) / R
    is_star_abs_err = np.abs(is_star_estimate - I_true)


    # 3. Importance Sampling with proposal N(theta_ce, 1) from CE.
    samples_theta_ce = rng.normal(loc=theta_ce, scale=1, size=R)
    weights_theta_ce = f0(samples_theta_ce) / f_theta(samples_theta_ce, theta_ce)
    is_ce_estimate = np.mean(weights_theta_ce * (samples_theta_ce > 4))
    is_ce_variance = np.var(weights_theta_ce * (samples_theta_ce > 4), ddof=1) / R
    is_ce_abs_err = np.abs(is_ce_estimate - I_true)

    # Build a results table using tabulate.
    results_table = [
        ["CMC (N(0,1))", f"{cmc_estimate:.6e}", f"{cmc_variance:.6e}", f"{cmc_abs_err:.6e}"],
        [f"IS with θ = {gamma_target:.2f}", f"{is_target_estimate:.6e}", f"{is_target_variance:.6e}", f"{is_abs_err:.6e}"],
        [f"IS with θ = {theta_ce:.4f} (CE)", f"{is_ce_estimate:.6e}", f"{is_ce_variance:.6e}", f"{is_ce_abs_err:.6e}"],
        [f"IS with θ = {theta_star:.4f} (opt)", f"{is_star_estimate:.6e}", f"{is_star_variance:.6e}", f"{is_star_abs_err:.6e}"]
    ]
    headers = ["Method", "Estimate", "Variance", "Absolute Error"]
    print("\nFinal Estimation Results for P(X > ", gamma_target,"):")
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

    # ---------------------------
    # Plot evolution (Figure 18)
    # ---------------------------

    # Plot for γ values
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(range(1, len(gamma_history) + 1), gamma_history, marker='o', linestyle='-')
    ax1.axhline(y=gamma_target, color='r', linestyle='--', label=f"Target γ = {gamma_target}")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    fig1_out = os.path.join(results_path, "ch5_cross_entropy_normal_1D_gammas.pdf")
    plt.savefig(fig1_out, bbox_inches='tight')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("γ (Quantile)")
    ax1.set_title("Evolution of γ Values")
    print("Saved γ evolution plot to", fig1_out)

    # Plot for θ values
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(range(len(theta_history)), theta_history, marker='o', linestyle='-')
    ax2.axhline(y=theta_star, color='r', linestyle='--', label=f"Theory Optimal θ = {np.round(theta_star, 4)}")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()
    fig2_out = os.path.join(results_path, "ch5_cross_entropy_normal_1D_thetas.pdf")
    plt.savefig(fig2_out, bbox_inches='tight')
    print("Saved θ evolution plot to", fig2_out)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("θ")
    ax2.set_title("Evolution of θ Values")
    plt.show()


if __name__ == "__main__":
    main()
