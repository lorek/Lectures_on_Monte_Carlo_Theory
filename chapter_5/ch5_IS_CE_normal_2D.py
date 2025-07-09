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
## File: ch5_IS_CE_normal_2D.py                                     ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Estimate P(X1>2, X2>2) for X ~ N([0,0], Σ) using crude Monte Carlo and importance sampling,
and plot the density contours with the target region shaded and special proposal points marked.
The proposal points (hard-coded) are [2,2], [2,2.83] and [2.4,3.8339].

Usage example:
    python estimate_bvn.py --R "100000" --results_path "results" --seed 31415
    python estimate_bvn.py --R "100000" --results_path "results" --seed None
"""

import argparse
import os
import numpy as np
from numpy.random import default_rng, PCG64
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from tabulate import tabulate



def ParseArguments():
    parser = argparse.ArgumentParser(description="Estimate P(X1>2, X2>2) using MC and importance sampling")
    parser.add_argument('--R', default="100000",
                        help="Total number of replications (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving plots (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None' for no fixed seed (default: %(default)s)")
    args = parser.parse_args()
    return args





def crude_monte_carlo(R, rng, mu, Sigma, threshold=2.0):
    """
    Crude Monte Carlo estimator for P(X1>2, X2>2) where X ~ N(mu, Sigma).
    """
    samples = rng.multivariate_normal(mu, Sigma, size=R)
    indicator = (samples[:, 0] > threshold) & (samples[:, 1] > threshold)
    prob_estimate = np.mean(indicator)
    variance_estimate = np.var(indicator, ddof=1) / R
    return prob_estimate, variance_estimate


def importance_sampling(R, rng, mu, Sigma, mu_proposal, threshold=2.0):
    """
    Importance sampling estimator for P(X1>2, X2>2) using proposal distribution N(mu_proposal, Sigma).
    """
    samples = rng.multivariate_normal(mu_proposal, Sigma, size=R)
    # Compute density ratios: f(x) / g(x)
    original_pdf = multivariate_normal.pdf(samples, mean=mu, cov=Sigma)
    proposal_pdf = multivariate_normal.pdf(samples, mean=mu_proposal, cov=Sigma)
    weights = original_pdf / proposal_pdf
    indicator = (samples[:, 0] > threshold) & (samples[:, 1] > threshold)
    estimate = np.mean(weights * indicator)
    variance_estimate = np.var(weights * indicator, ddof=1) / R
    return estimate, variance_estimate


def main():
    args = ParseArguments()
    R = int(args.R)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.strip().lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed=seed))

    # Define parameters for the bivariate normal distribution.
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, np.sqrt(2)], [np.sqrt(2), 4.0]])

    # Exact probability (for reference) can be computed using the CDF of the normal distribution.
    # Here we know (from your notes) that the exact value is approximately:
    I_true = 0.01861819992

    # Compute crude Monte Carlo estimator.
    crude_est, var_crude = crude_monte_carlo(R, rng, mu, Sigma, threshold=2.0)

    # Define the proposal points (hard-coded).
    proposals = {
        "mu = [2, 2]": np.array([2.0, 2.0]),
        "mu = [2, 2.83]": np.array([2.0, 2.83]),
        "mu = [2.4, 3.8339]": np.array([2.4, 3.8339])
    }

    # Compute importance sampling estimates for each proposal.
    is_results = {}
    for label, mu_prop in proposals.items():
        est, var_is = importance_sampling(R, rng, mu, Sigma, mu_prop, threshold=2.0)
        is_results[label] = (est, var_is)

    # Prepare a results table.
    headers = ["Method", "Estimate",  "Absolute Error", "Variance",]
    table = []
    # Crude MC results.
    table.append(["CMC", f"{crude_est:.8f}", f"{abs(crude_est - I_true):.8e}", f"{var_crude:.8e}" ])
    # Importance sampling results.
    for label, (est, var_is) in is_results.items():
        table.append([f"IS {label}", f"{est:.8f}", f"{abs(est - I_true):.8e}", f"{var_is:.8e}"])

    print("\n--- Simulation Results ---")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # ---------------------
    # Plot density contours and highlight the region A = {x: x1 > 2, x2 > 2}
    # ---------------------
    x1_vals = np.linspace(-3, 3, 300)
    x2_vals = np.linspace(-4.5, 4.5, 300)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    pos = np.dstack((X1, X2))
    density = multivariate_normal.pdf(pos, mean=mu, cov=Sigma)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Draw contour lines.
    contour = ax.contour(X1, X2, density, levels=10, colors='black', linewidths=1)
    # Shade the region where x1 > 2 and x2 > 2.
    # For simplicity, we use a polygon covering the rectangle [2, max(x1)] x [2, max(x2)]
    from matplotlib.patches import Rectangle
    rect = Rectangle((2, 2), x1_vals[-1] - 2, x2_vals[-1] - 2, color='gray', alpha=0.3)
    ax.add_patch(rect)






    # Plot special proposal points with specified colors.
    special_points = {
        "(0,0)": (0, 0, "black"),
        "(2,2)": (2, 2, "green"),
        "(2,2.83)": (2, 2.83, "blue"),
        "(2.4,3.8339)": (2.4, 3.8339, "red")
    }
    for label, (x, y, color) in special_points.items():
        ax.scatter(x, y, s=60, color=color, label=label)

    ax.legend()
    plt.grid(True)

    outfig = os.path.join(results_path, "ch5_IS_CE_normal_2D.pdf")
    plt.savefig(outfig, bbox_inches='tight', pad_inches=0)

    ax.set_title("Density contours of N(0,Σ) with shaded region (x1>2, x2>2)")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    print(f"\nPlot saved to: {outfig}")
    plt.show()


if __name__ == "__main__":
    main()
