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
## File: ch5_ratios_of_expectations.py                              ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Ratios of Expectations for π Estimation (Table V.6)

This script estimates the ratio
    I = (32 * ∫_0^1 [arctan(x)/(1+x^2)] dx) / (∫_0^1 exp(x) dx)
whose exact value is I_exact = π²/(e-1).

It compares two estimators:
  - The Crude Monte Carlo (CMC) estimator: uses independent U₁ and U₂.
  - The Ratio of Expectations (RoE) estimator: uses the same U for both parts.

For each given total number of replications R (which can be provided as a comma‐separated list),
the script computes:
  - hat{Y}^{CMC}and  hat{Y}^{RoE},
  - Their estimated variances,
  - The corresponding 95% error bounds b = 1.96\sqrt{Var},
  - The variance ratio Var(Y^{CMC})/Var(Y^{RoE}),
  - And the absolute errors compared to the exact value.
Results are displayed in a table (using tabulate).
"""

import numpy as np
import argparse
import os
from tabulate import tabulate
from scipy.stats import norm, chi2
from numpy.random import default_rng, PCG64

# Exact value: I_exact = π²/(e-1)
I_exact = (np.pi ** 2) / (np.e - 1)

def ParseArguments():
    parser = argparse.ArgumentParser(description="Ratios of Expectations for π Estimation")
    parser.add_argument('--R', default="10,100,1000,10000",
                        help="Comma-separated list of total replications (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving outputs (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG (default: 31415). Use 'None' (case-insensitive) for no fixed seed.")

    args = parser.parse_args()
    return args


def fun_k1(x):
    """Compute 32*arctan(x)/(1+x²)."""
    return 32 * np.arctan(x) / (1 + x ** 2)


def fun_k2(x):
    """Compute exp(x)."""
    return np.exp(x)





def cmc_estimator(R, rng):
    """
    Crude Monte Carlo estimator.
    Generates independent U1 and U2 of size R, then computes:
      z_CMC = mean(fun_k1(U1)) / mean(fun_k2(U2))
      and estimates variance as:
      Var_CMC = (1/(R² * (mean(fun_k2(U2)))²)) * Σ (fun_k1(U1) - z_CMC*fun_k2(U2))².
    Returns (z_CMC, Var_CMC, U1, U2).
    """
    U1 = rng.uniform(0, 1, R)
    U2 = rng.uniform(0, 1, R)
    X1 = fun_k1(U1)
    X2 = fun_k2(U2)
    z_CMC = np.mean(X1) / np.mean(X2)
    Var_CMC = np.sum((X1 - z_CMC * X2) ** 2) / (R ** 2 * (np.mean(X2) ** 2))
    return z_CMC, Var_CMC, U1, U2


def roe_estimator(R, rng):
    """
    Ratio of Expectations (RoE) estimator.
    Uses the same U for both components.
    Returns (z_RoE, Var_RoE, U).
    """
    U = rng.uniform(0, 1, R)
    Y1 = fun_k1(U)
    Y2 = fun_k2(U)
    z_RoE = np.mean(Y1) / np.mean(Y2)
    Var_RoE = np.sum((Y1 - z_RoE * Y2) ** 2) / (R ** 2 * (np.mean(Y2) ** 2))
    return z_RoE, Var_RoE, U


def main():
    args = ParseArguments()
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)
    # Process R as a list of integers.
    R_list = [int(x.strip()) for x in args.R.split(",")]

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    # Prepare table rows for each R.
    # Rows: Metric names; Columns: one per R.
    header = ["Metric"] + [f"R={R_val}" for R_val in R_list]

    # Initialize rows as lists
    row_Y_CMC = ["hat{Y}^{CMC}"]
    row_Y_RoE = ["hat{Y}^{RoE}"]
    row_AbsError_CMC = ["|\\hat{Y}^{CMC}-I|"]
    row_AbsError_RoE = ["|\\hat{Y}^{RoE}-I|"]
    row_Var_CMC = ["widehat{Var}(Y^{CMC})"]
    row_Var_RoE = ["widehat{Var}(Y^{RoE})"]
    row_b_CMC = ["b^{CMC}=1.96\\sqrt{Var}"]
    row_b_RoE = ["b^{RoE}=1.96\\sqrt{Var}"]
    row_VarRatio = ["VarRatio"]


    for R_val in R_list:
        # Compute estimators.
        z_roe, var_roe, _ = roe_estimator(R_val, rng)
        z_cmc, var_cmc, _, _ = cmc_estimator(R_val, rng)
        b_cmc = 1.96 * np.sqrt(var_cmc)
        b_roe = 1.96 * np.sqrt(var_roe)
        abs_err_cmc = abs(z_cmc - I_exact)
        abs_err_roe = abs(z_roe - I_exact)
        var_ratio = var_cmc / var_roe if var_roe != 0 else np.nan

        row_Y_CMC.append(f"{z_cmc:.5f}")
        row_Y_RoE.append(f"{z_roe:.5f}")
        row_AbsError_CMC.append(f"{abs_err_cmc:.5f}")
        row_AbsError_RoE.append(f"{abs_err_roe:.5f}")
        row_Var_CMC.append(f"{var_cmc:.5f}")
        row_Var_RoE.append(f"{var_roe:.5f}")
        row_b_CMC.append(f"{b_cmc:.5f}")
        row_b_RoE.append(f"{b_roe:.5f}")
        row_VarRatio.append(f"{var_ratio:.5f}")


    table = tabulate([row_Y_CMC, row_Y_RoE, row_AbsError_CMC, row_AbsError_RoE, row_Var_CMC, row_Var_RoE,
                      row_b_CMC, row_b_RoE, row_VarRatio],
                     headers=header, tablefmt="grid")
    print("\n=== Simulation Results   ===")
    print(table)




if __name__ == "__main__":
    main()
