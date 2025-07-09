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
## File: ch5_collective_risk_Erlang_stratified.py                   ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




""""
Estimate the ruin probability
    I = P(S > 32)
in the collective risk model. Claims are Erlang(4, 2.2) (i.e., sum of 4 Exponential(2.2))
and the number of claims is Poisson(λ) with λ=3.3. For N=0, we set S=0.

We do both a CMC estimator and a stratified estimator (stratifying on N).
We accept m as a comma-separated list (default "3,4,5,6,7") and run stratified
sampling for each m. We then:
  1) Print a "Detailed Table for m=largest" showing R_i, p_i, etc.
  2) Print a "Final Results Table" with columns = CMC, m=3, m=4, …, listing:
       - Estimate
       - b (1.96×√Var)
       - Var \(Y^{\rm CMC}\)/Var \(Y\)
"""

import argparse
import numpy as np
from scipy.stats import poisson
from scipy.special import factorial
from tabulate import tabulate




def ParseArguments():
    parser = argparse.ArgumentParser(description="Estimate ruin probability in the collective risk model")
    parser.add_argument('--R', default="10000",
                        help="Total number of replications (default: %(default)s)")
    parser.add_argument('--lambd', default="3.3",
                        help="Poisson parameter λ for number of claims (default: %(default)s)")
    parser.add_argument('--shape', default="4",
                        help="Shape parameter for Erlang distribution (default: %(default)s)")
    parser.add_argument('--scale', default="2.2",
                        help="Scale parameter for Erlang distribution (default: %(default)s)")
    parser.add_argument('--t', default="32",
                        help="Threshold t for S (default: %(default)s)")
    parser.add_argument('--m', default="3,4,5,6,7",
                        help="Comma-separated list of strata counts (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PCG64 RNG (default: %(default)s)")
    args = parser.parse_args()
    return args


def simulate_CMC(R, lambd, shape, scale, thres, rng):
    """Crude Monte Carlo simulation for I = P(S > thres)."""
    indicators = np.zeros(R)
    totals = np.zeros(R)
    for i in range(R):
        N = rng.poisson(lambd)
        if N > 0:
            # Each claim ~ Erlang(shape, scale) => sum of 'shape' exponentials
            claims = [np.sum(rng.exponential(scale, size=int(shape))) for _ in range(N)]
            S = np.sum(claims)
        else:
            S = 0
        totals[i] = S
        indicators[i] = 1 if S > thres else 0
    prob = np.mean(indicators)
    var_est = np.var(indicators, ddof=1) / R
    avg_total = np.mean(totals)
    return prob, var_est, avg_total


def get_proportional_allocation(R, lambd, m):
    """
    For stratification on N, define strata:
       A_j = {N = j} for j=0,...,m-2, and A_{m-1} = {N >= m-1}.
    Allocate replications proportionally, except for R_0=1.
    Returns:
       R_vec: allocations for each stratum (length m)
       p:     probabilities p_j
    """
    R_vec = np.zeros(m, dtype=int)
    p = []
    for j in range(m - 1):
        p_j = np.exp(-lambd) * (lambd ** j) / factorial(j)
        p.append(p_j)
        R_vec[j] = int(np.ceil(R * p_j))
    # fix stratum 0
    R_vec[0] = 1
    p_sum = np.sum(p)
    p_last = 1 - p_sum
    p.append(p_last)  # now p has length m
    R_vec[m - 1] = R - np.sum(R_vec[:m - 1])
    return R_vec, np.array(p)


def simulate_stratum(j, Rj, lambd, shape, scale, thres, m, rng):
    """
    Simulate Rj replications for stratum j.
      - j=0 => N=0 => S=0 => indicator=0
      - j=1,...,m-2 => N=j
      - j=m-1 => N >= m-1, simulated by inversion
    Returns array of 0/1 indicators.
    """
    indicators = np.zeros(Rj)
    if j == 0:
        return indicators
    elif j < (m - 1):
        # stratum j => N=j
        for i in range(Rj):
            S = 0
            for _ in range(j):
                claim = np.sum(rng.exponential(scale, size=int(shape)))
                S += claim
            indicators[i] = 1 if S > thres else 0
    else:
        # last stratum => N >= m-1
        rho = poisson.cdf(m - 2, lambd)
        for i in range(Rj):
            U = rng.random()
            N = poisson.ppf(rho + (1 - rho) * U, lambd)
            N = int(N)
            S = 0
            for _ in range(N):
                claim = np.sum(rng.exponential(scale, size=int(shape)))
                S += claim
            indicators[i] = 1 if S > thres else 0
    return indicators


def simulate_stratified(R, lambd, shape, scale, thres, m, rng):
    """Perform stratified simulation with proportional allocation."""
    R_vec, weights = get_proportional_allocation(R, lambd, m)
    stratum_means = np.zeros(m)
    stratum_vars = np.zeros(m)
    for j in range(m):
        indicators = simulate_stratum(j, R_vec[j], lambd, shape, scale, thres, m, rng)
        if R_vec[j] > 1:
            stratum_means[j] = np.mean(indicators)
            stratum_vars[j] = np.var(indicators, ddof=1)
        else:
            stratum_means[j] = indicators[0] if R_vec[j] == 1 else 0
            stratum_vars[j] = 0
    I_strat = np.sum(weights * stratum_means)
    var_strat = np.sum(weights ** 2 * stratum_vars / R_vec)
    return I_strat, var_strat, weights, R_vec, stratum_means, stratum_vars


def main():
    args = ParseArguments()
    R = int(args.R)
    lambd = float(args.lambd)
    shape = float(args.shape)
    scale = float(args.scale)
    thres = float(args.t)
    # parse the comma-separated list of m
    m_list = [int(x) for x in args.m.split(',')]
    # sort to identify largest m
    m_list.sort()

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))


    # -------------------------
    # 1) Crude Monte Carlo
    # -------------------------
    I_cmc, var_cmc, avg_total = simulate_CMC(R, lambd, shape, scale, thres, rng)
    err_cmc = 1.96 * np.sqrt(var_cmc)

    print(f"ES (theoretical) = {scale * shape * lambd:.4f}")
    print(f"Estimating P(S > {thres:.0f}) for S = X₁ + ... + X_N,")
    print(f"  where Xᵢ ~ Erlang({int(shape)},{scale}) and N ~ Poisson({lambd})")
    print(f"Using Crude Monte Carlo and stratified sampling with strata m = {args.m}\n")

    # -------------------------
    # 2) Stratified for each m
    # -------------------------
    results = {}
    for mm in m_list:
        I_strat, var_strat, weights, R_vec, stratum_means, stratum_vars = simulate_stratified(
            R, lambd, shape, scale, thres, mm, rng
        )
        err_strat = 1.96 * np.sqrt(var_strat)
        ratio = var_cmc / var_strat if var_strat > 0 else np.nan
        results[mm] = (I_strat, var_strat, err_strat, ratio, weights, R_vec, stratum_means, stratum_vars)

    # 2a) Print the "Detailed Table for m = largest_m" FIRST
    largest_m = m_list[-1]
    I_strat, var_strat, err_strat, ratio, weights, R_vec, stratum_means, stratum_vars = results[largest_m]

    print(f"--- Detailed Table for m={largest_m} ---")
    # Build a table for i=0,...,largest_m-2 and i=largest_m-1
    # Rows: i, R_i, p_i, mean(Ind), std(Ind)
    row_i = ["i"]
    row_R = ["R_i"]
    row_p = ["p_i"]
    for j in range(largest_m):
        if j == largest_m - 1:
            row_i.append(f"{j}\n(N≥{j})")
        else:
            row_i.append(str(j))
        row_p.append(f"{weights[j]:.4f}")
        row_R.append(str(R_vec[j]))



    detailed_data = [row_i, row_p, row_R]
    print(tabulate(detailed_data, tablefmt="grid" ,stralign="center", numalign="center"))
    print()

    # -------------------------
    # 2b) Build the Final Results Table
    #     with columns = CMC, m=3, m=4, ...
    # -------------------------
    print("--- Final Results Table ---")

    headers = ["", "CMC"] + [f"m={mm}" for mm in m_list]
    # Build 3 rows:
    row_est = ["Estimate:"]
    row_b = ["b (1.96*sqrt(Var)):"]
    row_var_ratio = ["Var Y^CMC/Var Y:"]

    # Fill in CMC columns
    row_est.append(f"{I_cmc:.5f}")
    row_b.append(f"{err_cmc:.5f}")
    row_var_ratio.append("--")

    # Fill in columns for each m
    for mm in m_list:
        I_strat, var_strat, err_strat, ratio, _, _, _, _ = results[mm]
        row_est.append(f"{I_strat:.5f}")
        row_b.append(f"{err_strat:.5f}")
        row_var_ratio.append(f"{ratio:.6f}")

    final_table = [row_est, row_b, row_var_ratio]
    print(tabulate(final_table, headers=headers, tablefmt="grid", stralign="center", numalign="center"))
    print()


if __name__ == "__main__":
    main()
