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
## File: ch5_european_option_stratified.py                          ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Estimate the price of a European call option using both Crude Monte Carlo (CMC)
and stratified sampling as in Example 2.5. The Black–Scholes price is computed for reference.
Stratification is performed for different numbers of strata (provided as a comma-separated list).

Usage example:
    python stratified_option_sim.py --R 100000 --Rpilot 10000 --m "5,10,50,100,200" --K 100 --S0 100 --r 0.05 --sigma 0.25 --seed 31415
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tabulate import tabulate


def ParseArguments():
    parser = argparse.ArgumentParser(description="Estimate European call option price using stratified sampling")
    parser.add_argument('--R', default="100000",
                        help="Total number of replications (default: %(default)s)")
    parser.add_argument('--Rpilot', default="10000",
                        help="Number of pilot replications (default: %(default)s)")
    parser.add_argument('--m', default="5,10,50,100,200",
                        help="Comma-separated list of numbers of strata (default: %(default)s)")
    parser.add_argument('--K', default="100",
                        help="Strike price K (default: %(default)s)")
    parser.add_argument('--S0', default="100",
                        help="Initial stock price S(0) (default: %(default)s)")
    parser.add_argument('--r', default="0.05",
                        help="Interest rate r (default: %(default)s)")
    parser.add_argument('--sigma', default="0.25",
                        help="Volatility sigma (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PCG64 random number generator. Use 'None' for no fixed seed (default: %(default)s)")
    args = parser.parse_args()
    return args




def simulate_strata(m, R, weights, rng):
    """
    For a given number of strata m, simulate R replications using stratified sampling.
    Returns:
      - N_rvs: simulated N(0,1) random variates (concatenated over strata)
      - strata_stds: array of standard deviations of N(0,1) within each stratum (for info)
      - Y_i_strata: array with stratum index for each simulated point
      - U_all: concatenated uniform samples used in simulation (for potential CMC sample)
    """
    strata_stds = np.zeros(m)
    N_rvs = []
    Y_i_strata = []
    U_all = []

    # Loop over strata
    for i in range(m):
        # Number of replications allocated to stratum i
        ni = int(weights[i] * R)
        if ni < 1:
            ni = 1  # ensure at least one simulation per stratum
        # Generate ni independent U(0,1) samples using rng
        U = rng.random(ni)
        U_all.append(U)
        # Map U into the subinterval for stratum i: (i/m, (i+1)/m]
        V = i / m + (1 / m) * U
        # Inverse transform: simulate N(0,1) conditional on strata
        dane = norm.ppf(V)
        strata_stds[i] = np.std(dane, ddof=1)
        N_rvs.append(dane)
        Y_i_strata.extend([i] * ni)
    # Concatenate arrays for output
    N_rvs = np.concatenate(N_rvs)
    U_all = np.concatenate(U_all)
    Y_i_strata = np.array(Y_i_strata)

    return N_rvs, strata_stds, Y_i_strata, U_all


def run_pilot(m, Rpilot, rng):
    """
    Run a pilot simulation with Rpilot replications equally allocated over m strata.
    Returns the estimated standard deviations per stratum.
    """
    weights_uniform = np.ones(m) / m
    _, pilot_stds, _, _ = simulate_strata(m, Rpilot, weights_uniform, rng)
    return pilot_stds


def compute_option_price(S0, r, sigma, K):
    """
    Compute the exact Black-Scholes price for a European call option.
    """
    mu = r - sigma ** 2 / 2
    d1 = (np.log(S0 / K) + r + sigma ** 2 / 2) / sigma
    d2 = d1 - sigma
    price = S0 * norm.cdf(d1) - K * np.exp(-r) * norm.cdf(d2)
    return price, d1, d2


def main():
    args = ParseArguments()
    R = int(args.R)
    Rpilot = int(args.Rpilot)
    K = float(args.K)
    S0 = float(args.S0)
    r = float(args.r)
    sigma = float(args.sigma)
    # Parse the comma-separated list of strata numbers
    m_list = [int(x) for x in args.m.split(",")]

    # Create output folder (if needed)
    results_path = "results"
    os.makedirs(results_path, exist_ok=True)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    # Compute exact option price using Black-Scholes formula
    exact_price, d1, d2 = compute_option_price(S0, r, sigma, K)
    print("\nExact Black–Scholes call option price: {:.5f}".format(exact_price))
    print("Parameters: S0 = {}, K = {}, r = {}, sigma = {}".format(S0, K, r, sigma))

    # -------------------------
    # Crude Monte Carlo Simulation
    # -------------------------
    U_cmc = rng.random(R)
    B_cmc = norm.ppf(U_cmc)
    S1_cmc = S0 * np.exp((r - sigma ** 2 / 2) + sigma * B_cmc)
    payoff_cmc = np.exp(-r) * np.maximum(S1_cmc - K, 0)

    cmc_mean = np.mean(payoff_cmc)
    cmc_var = np.var(payoff_cmc, ddof=1) / R
    cmc_error = 1.96 * np.sqrt(cmc_var)
    print("\n[CMC] Estimated Price: {:.5f}, Absolute Error: {:.4e}".format(cmc_mean, abs(cmc_mean - exact_price)))
    print("[CMC] Variance: {:.4e}, 95% CI half-width: {:.4e}".format(cmc_var, cmc_error))

    # -------------------------
    # Stratified Sampling for different m values
    # -------------------------
    strat_results = {}  # key: m, value: (strat_est, strat_error, strat_var)
    for m in m_list:

        print("Performing  simulation for m = {} strata with Rpilot = {}...".format(m, Rpilot))

        pilot_stds = run_pilot(m, Rpilot, rng)

        # Compute new weights from pilot std estimates (replace any zero with a small number)
        weights = pilot_stds.copy()
        weights[weights == 0] = 1.0 / R
        weights = weights / np.sum(weights)

        N_strata, strata_stds, Y_i_strata, U_all = simulate_strata(m, R, weights, rng)
        S1_strata = S0 * np.exp((r - sigma ** 2 / 2) + sigma * N_strata)
        payoff_strata = np.exp(-r) * np.maximum(S1_strata - K, 0)

        # Calculate the stratum-wise mean and variance.
        means = np.zeros(m)
        vars_ = np.zeros(m)
        counts = np.zeros(m, dtype=int)
        for i in range(m):
            mask = (Y_i_strata == i)
            counts[i] = np.sum(mask)
            if counts[i] > 0:
                means[i] = np.mean(payoff_strata[mask])
                vars_[i] = np.var(payoff_strata[mask], ddof=1)
            else:
                means[i] = 0
                vars_[i] = 0
        # The stratified estimator: since each stratum has probability 1/m
        strat_est = np.sum(means) / m
        # Variance: sum_{i=0}^{m-1} (1/m)^2 * variance_i / n_i
        strat_var = np.sum((1 / m) ** 2 * vars_ / counts)
        strat_error = 1.96 * np.sqrt(strat_var)
        strat_results[m] = (strat_est, strat_error, strat_var)

    # -------------------------
    # Build final table:
    # Rows:
    #   1. "Estimate:" -> CMC and each stratified estimator
    #   2. "b (1.96*sqrt(Var)):" -> confidence interval half-width for CMC and stratified
    #   3. "Var Y^CMC/Var Y:" -> ratio of CMC variance to stratified variance
    # Columns: CMC, then one column per m (labeled m=5, m=10, etc.)
    # -------------------------
    headers = [""]  # first empty header
    headers.append("CMC")
    for m in m_list:
        headers.append("m={}".format(m))

    row_estimate = ["Estimate:"]
    row_b = ["b (1.96*sqrt(Var)):"]
    row_ratio = ["Var Y^CMC/Var Y:"]
    row_abs_error = ["Abs error"]

    row_estimate.append("{:.5f}".format(cmc_mean))
    row_b.append("{:.4e}".format(cmc_error))
    row_ratio.append("1.0000")  # for CMC the ratio is 1
    row_abs_error.append("{:.5f}".format(np.abs(cmc_mean -exact_price )))

    for m in m_list:
        strat_est, strat_error, strat_var = strat_results[m]
        row_estimate.append("{:.5f}".format(strat_est))
        row_b.append("{:.2e}".format(strat_error))
        ratio = cmc_var / strat_var if strat_var > 0 else float('nan')
        row_ratio.append("{:.5f}".format(ratio))
        row_abs_error.append("{:.5f}".format(np.abs(strat_est - exact_price)))

    final_table = [row_estimate, row_abs_error, row_b, row_ratio,]

    print("\n--- Final Results Table ---")
    print(tabulate(final_table, headers=headers, tablefmt="grid", stralign="left", numalign="left"))


if __name__ == "__main__":
    main()
