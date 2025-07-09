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
## File: ch5_ContrVariates_normal_1D.py                             ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Control Variate for Estimating P(Z > z_threshold)

We wish to estimate I = P(Z > z_threshold) for Z ~ N(0,1). Using the inverse
transform method we have Z = Φ⁻¹(U) for U ~ U(0,1), so that
I = P(Z > z_threshold) = P(U > Φ(z_threshold)).

A crude Monte Carlo (CMC) estimator is given by
    Ŷ_CMC = (1/R) ∑ 1(U_i > Φ(z_threshold)),
and its variance can be computed directly.

To reduce variance, we choose a control variate
    X = 1(U > beta),
with a fixed beta (which should be chosen close to Φ(z_threshold)). We then
estimate the optimal coefficient c via a pilot simulation. The control variate
estimator is then
    Ŷ_CV = Y + ĉ (X - E[X]).
"""

import numpy as np
import argparse
import os
from scipy.stats import norm, chi2
from numpy.random import default_rng, PCG64
from tabulate import tabulate


def ParseArguments():
    parser = argparse.ArgumentParser(description="Control Variate for P(Z > z_threshold)")
    parser.add_argument('--z_threshold', default="4.0", help="Threshold for Z (default: %(default)s)")
    parser.add_argument('--beta', default="0.9999", help="Control variate threshold (default: %(default)s)")
    parser.add_argument('--Rpilot', default="1000000", help="Number of pilot replications (default: %(default)s)")
    parser.add_argument('--R', default="10000000", help="Number of main replications (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Folder for saving outputs (default: %(default)s)")
    return parser.parse_args()


def main():
    args = ParseArguments()
    # Handle seed: if user specifies "None" (case-insensitive), then seed is None.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed=seed))
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Parse simulation parameters.
    z_threshold = float(args.z_threshold)
    beta = float(args.beta)
    Rpilot = int(args.Rpilot)
    R = int(args.R)

    # Compute the threshold for the indicator functions.
    alpha = norm.cdf(z_threshold)  # P(Z <= z_threshold)
    p_true = 1 - alpha  # Actual probability P(Z > z_threshold)
    E_X = 1 - beta  # Control variate expected value

    # ------------------------------
    # Pilot Simulation to estimate c
    # ------------------------------
    U_pilot = rng.uniform(0, 1, Rpilot)

    Y_pilot = (U_pilot > alpha).astype(float)
    X_pilot = (U_pilot > beta).astype(float)

    Y_bar_pilot = np.mean(Y_pilot)
    X_bar_pilot = np.mean(X_pilot)
    cov_YX = np.mean((Y_pilot - Y_bar_pilot) * (X_pilot - X_bar_pilot))
    var_X = np.mean((X_pilot - X_bar_pilot) ** 2)

    # Estimate the control variate coefficient.
    c_hat = -cov_YX / var_X

    # -------------------------------
    # Main Simulation
    # -------------------------------
    U_main = rng.uniform(0, 1, R)
    Y = (U_main > alpha).astype(float)
    X = (U_main > beta).astype(float)
    Y_CV = Y + c_hat * (X - E_X)

    # Compute the CMC estimator variance:
    var_Y = np.var(Y, ddof=1)  # sample variance of Y
    var_CMC = var_Y / R  # Var(Y)/R

    # Compute the sample correlation between Y and X
    rho_sample = np.corrcoef(Y, X)[0, 1]

    # Theoretical variance for the control variate estimator is reduced by a factor (1 - rho^2)
    var_CV_theoretical = var_CMC * (1 - rho_sample ** 2)

    # Now compute the standard errors based on these theoretical variances.
    se_CMC = 1.96 * np.sqrt(var_CMC)
    se_CV = 1.96 * np.sqrt(var_CV_theoretical)

    Y_bar_CMC = np.mean(Y)
    Y_bar_CV = np.mean(Y_CV)

    abs_err_CMC = abs(Y_bar_CMC - p_true)
    abs_err_CV = abs(Y_bar_CV - p_true)


    # print("Var_CMC:", var_CMC)
    # print("Theoretical Var_CV (Var_CMC*(1-rho^2)):", var_CV_theoretical)
    # print("Standard Error CMC:", se_CMC)
    # print("Standard Error CV:", se_CV)


    # -------------------------------
    # Display the Results using Tabulate
    # -------------------------------
    table = [
        ["CMC", f"{Y_bar_CMC:.4e}", f"{var_CMC:.4e}", f"{np.sqrt(var_CMC):.4e}", f"{abs_err_CMC:.4e}"],
        ["CV", f"{Y_bar_CV:.4e}", f"{var_CV_theoretical:.4e}", f"{np.sqrt(var_CV_theoretical):.4e}",
         f"{abs_err_CV:.4e}"]
    ]
    headers = ["Method", "Estimate", "Var", "Std", "Absolute Error"]
    print("\n--- Simulation Results ---")
    print(tabulate(table, headers=headers, tablefmt="grid"))

    # Also print other computed quantities.
    print(f"\nz_threshold: {z_threshold}")
    print(f"Φ(z_threshold): {alpha:.8f}")
    print("Sample correlation rho:", rho_sample, ", rho^2 = ", rho_sample**2)
    print(f"Actual P(Z > {z_threshold}): {p_true:.8f}")
    print(f"Control variate coefficient c: {c_hat:.6f}")
    print(f"The variance was reduced by a factor of 1-rho^2 = {(1-rho_sample**2):.6f} times")
    # Save the results to a file if needed (optional)
    # For example, write results to a text file in results_path.

    # End of main simulation.


if __name__ == "__main__":
    main()
