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
## File: ch5_asian_option_stratified.py                             ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Stratified Sampling for a Bivariate Option Pricing Problem (Multiple m, Multiple K).

We simulate a bivariate standard normal vector using radial (inverse Rayleigh) strata.
Then transform via Cholesky to get (B(0.5), B(1)), which in turn define:
  S(1) = S(0)*exp(mu + sigma*B(1))
  S(0.5) = S(0)*exp(0.5*mu + sigma*B(0.5))
  A = (S(0.5)+S(1))/2
and payoff = e^{-r} * max(A - K, 0).

We allow multiple m-values (default: 5,10,50,100,200) and multiple K-values (default: 100,125,150,175).
For each m:
  - Perform a pilot simulation with Rpilot replications (uniform allocation).
  - For each K, compute pilot standard deviations and derive new allocations for R.
  - Perform main simulation, compute both CMC and stratified estimates.

We produce the original pilot plots (fig1, fig1b, fig2, fig2b, fig2c) plus a main-simulation scatter plot (fig3)
**only for the smallest m**. The final table has columns: [CMC, m=5, m=10, …], and each K occupies two rows:
  one for the estimate, one for the half-width b.

Run example:
    python multi_strat.py --R 100000 --Rpilot 10000 --m 5,10,50,100,200 \
        --K 100,125,150,175 --S0 100 --r 0.05 --sigma 0.25 \
        --results_path results --seed 31415
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from numpy.random import default_rng, PCG64
from tabulate import tabulate
import matplotlib.cm as cm


def ParseArguments():
    parser = argparse.ArgumentParser(description="Stratified sampling with multiple m and K.")
    parser.add_argument('--R', default="100000", help="Total number of replications (main simulation)")
    parser.add_argument('--Rpilot', default="10000", help="Number of pilot replications")
    parser.add_argument('--m', default="5,10,50,100,200",
                        help="Comma-separated list of strata counts (default: 5,10,50,100,200)")
    parser.add_argument('--K', default="100,125,150,175",
                        help="Comma-separated list of strike prices (default: 100,125,150,175)")
    parser.add_argument('--S0', default="100", help="Initial stock price S(0)")
    parser.add_argument('--r', default="0.05", help="Interest rate r")
    parser.add_argument('--sigma', default="0.25", help="Volatility sigma")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving plots (default: results)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for PCG64 RNG (use 'None' for no fixed seed).")
    return parser.parse_args()


def inverse_rayleigh(t):
    return np.sqrt(-2.0 * np.log(1.0 - t))


def simulate_bivariate_strata(m, vecR, rng):
    """
    Stratify a 2D standard normal distribution using radial strata (inverse Rayleigh).
    For stratum i=0,...,m-1, we allocate vecR[i] points:
      U1 ~ U(0,1), U2 ~ U(0,1)
      θ = 2π * U2
      V = i/m + (1/m)*U1
      Rr = inverse_rayleigh(V)
      X = Rr sin(θ), Y = Rr cos(θ)
    Returns:
      points: (N,2) array, strata_indices: array of length N
    """
    points = []
    strata_indices = []
    for i in range(m):
        ni = vecR[i]
        U1 = rng.random(ni)
        U2 = rng.random(ni)
        theta = 2.0 * np.pi * U2
        V = i / m + (1.0 / m) * U1
        Rr = inverse_rayleigh(V)
        X = Rr * np.sin(theta)
        Y = Rr * np.cos(theta)
        for j in range(ni):
            points.append([X[j], Y[j]])
            strata_indices.append(i)
    return np.array(points), np.array(strata_indices, dtype=int)


def main():
    args = ParseArguments()
    R = int(args.R)
    Rpilot = int(args.Rpilot)
    m_list = [int(mm) for mm in args.m.split(',')]
    m_list.sort()
    K_list = [float(kk) for kk in args.K.split(',')]
    K_list.sort()

    S0 = float(args.S0)
    r = float(args.r)
    sigma = float(args.sigma)
    mu = r - sigma ** 2 / 2

    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    seed = None if args.seed.strip().lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed=seed))

    # Covariance => (B(0.5), B(1))
    cov_matrix = np.array([[0.5, 0.5],
                           [0.5, 1.0]])
    A_chol = np.linalg.cholesky(cov_matrix).T

    # We'll store final results in: results[k_val][m] = (CMC_mean, CMC_b, strat_mean, strat_b)
    results = {k_val: {} for k_val in K_list}

    # We'll produce all the old pilot plots + main scatter only for the smallest m
    m_small = m_list[0]

    for m in m_list:
        print("Performing   simulations for m=", m , " ...")
        # PILOT
        # uniform allocation
        vecR_pilot = np.full(m, Rpilot // m, dtype=int)
        while vecR_pilot.sum() < Rpilot:
            for i in range(m):
                if vecR_pilot.sum() >= Rpilot:
                    break
                vecR_pilot[i] += 1

        pilot_points, pilot_strata = simulate_bivariate_strata(m, vecR_pilot, rng)
        B05_pilot, B1_pilot = np.dot(pilot_points, A_chol).T
        S1_pilot = S0 * np.exp(mu + sigma * B1_pilot)
        S05_pilot = S0 * np.exp(0.5 * mu + sigma * B05_pilot)
        A_pilot = (S05_pilot + S1_pilot) / 2

        # If m == m_small, produce all original pilot plots:
        if m == m_small:
            fig1b, ax1b = plt.subplots( figsize=(8,8))
            fig1, ax1 = plt.subplots( figsize=(8,8))
            fig2, ax2 = plt.subplots( figsize=(8,8))
            fig2b, ax2b = plt.subplots( figsize=(8,5))


            N05_pilot, N1_pilot = pilot_points.T
            phis = np.linspace(0, 2 * np.pi, 500)

            lin = np.linspace(0.1, 0.92, m)
            permuted_lin = rng.permutation(lin)
            colors = cm.rainbow(permuted_lin)




            for i in range(m):
                mask = (pilot_strata == i)
                # boundary circle
                cir_r = inverse_rayleigh((i + 1) / m)
                cir_x = cir_r * np.cos(phis)
                cir_y = cir_r * np.sin(phis)
                # transform boundary
                cir_B_x, cir_B_y = np.dot(np.column_stack((cir_x, cir_y)), A_chol).T
                cir_S_x = S0 * np.exp(mu + sigma * cir_B_x)
                cir_S_y = S0 * np.exp(mu + sigma * cir_B_y)

                ax1.scatter(B05_pilot[mask], B1_pilot[mask], s=0.5, color=colors[i], label=f"Stratum {i}")
                ax1.scatter(cir_B_x, cir_B_y, s=0.2, color='gray')

                ax1b.scatter(N05_pilot[mask], N1_pilot[mask], s=0.5, color=colors[i])
                ax1b.scatter(cir_x, cir_y, s=0.2, color='gray')

                ax2.scatter(S05_pilot[mask], S1_pilot[mask], s=0.5, color=colors[i])
                ax2.scatter(cir_S_x, cir_S_y, s=0.2, color='gray')

                ax2b.scatter(np.where(mask), A_pilot[mask], s=0.5, color=colors[i])
                ax2b.axhline(y=K_list[0], color='gray', linewidth=0.5)



            fig1.savefig(os.path.join(results_path, "ch5_asian_option_stratified_normal_2D_Sigma.pdf"), bbox_inches="tight")
            fig1b.savefig(os.path.join(results_path, "ch5_asian_option_stratified_normal_2D_standard.pdf"), bbox_inches="tight")
            fig2.savefig(os.path.join(results_path, "ch5_asian_option_stratified_S05_S1.pdf"), bbox_inches="tight")
            fig2b.savefig(os.path.join(results_path, "ch5_asian_option_stratified_points_A.pdf"), bbox_inches="tight")

            ax1.set_title("B05 vs B1")
            ax1b.set_title("Original coords")
            ax2.set_title("S(0.5) vs S(1)")
            ax2b.set_title("A=(S(0.5)+S(1))/2")


        # For each K, compute pilot std
        pilot_stds_dict = {}
        for k_val in K_list:
            stdevs = np.zeros(m)
            for i in range(m):
                mask = (pilot_strata == i)
                if mask.sum() > 0:
                    payoff_pilot = np.exp(-r) * np.maximum(0, A_pilot[mask] - k_val)
                    stdevs[i] = payoff_pilot.std()
            pilot_stds_dict[k_val] = stdevs

        # MAIN simulation
        for k_val in K_list:
            stds = pilot_stds_dict[k_val]
            stds_adj = np.where(stds == 0, 1.0 / R, stds)
            weights = stds_adj / stds_adj.sum()
            vecR_main = (weights * R).astype(int)

            # Ensure each stratum has at least one sample
            vecR_main[vecR_main == 0] = 1

            # If the total allocation exceeds R, subtract samples from strata with more than one sample until sum equals R
            while vecR_main.sum() > R:
                # Find a stratum with more than one sample; you could choose the one with the largest allocation
                idx = np.argmax(vecR_main)
                if vecR_main[idx] > 1:
                    vecR_main[idx] -= 1
                else:
                    break  # Should not happen if at least one stratum has >1 sample

            # 1) Generate R i.i.d. points for CMC
            X_cmc = rng.normal(size=(R, 2))
            B05_cmc, B1_cmc = np.dot(X_cmc, A_chol).T
            S1_cmc = S0 * np.exp(mu + sigma * B1_cmc)
            S05_cmc = S0 * np.exp(0.5 * mu + sigma * B05_cmc)
            A_cmc = (S05_cmc + S1_cmc) / 2
            payoff_cmc = np.exp(-r) * np.maximum(0, A_cmc - k_val)
            cmc_mean = payoff_cmc.mean()
            cmc_var = payoff_cmc.var(ddof=1) / R
            cmc_b = 1.96 * np.sqrt(cmc_var)

            # 2) Stratified
            main_points, main_strata = simulate_bivariate_strata(m, vecR_main, rng)
            B05_main, B1_main = np.dot(main_points, A_chol).T
            S1_main = S0 * np.exp(mu + sigma * B1_main)
            S05_main = S0 * np.exp(0.5 * mu + sigma * B05_main)
            A_main = (S05_main + S1_main) / 2
            payoff_main = np.exp(-r) * np.maximum(0, A_main - k_val)

            means = np.zeros(m)
            vars_ = np.zeros(m)
            for i in range(m):
                mask = (main_strata == i)

                count = mask.sum()
                if count > 1:
                    arr_i = payoff_main[mask]
                    means[i] = arr_i.mean()
                    vars_[i] = arr_i.var(ddof=1)
                elif count == 1:
                    arr_i = payoff_main[mask]
                    means[i] = arr_i[0]
                    vars_[i] = 0.0

            strat_mean = means.mean()
            strat_var = np.sum((1.0 / m) ** 2 * vars_ / vecR_main)
            strat_b = 1.96 * np.sqrt(strat_var)

            results[k_val][m] = (cmc_mean, cmc_b, strat_mean, strat_b)



    # Build final table:
    # columns = [CMC, m=5, m=10, ...], 2 rows per K (estimate, b)
    # We'll pick the CMC from the run with the smallest m for each K, to have a single reference CMC.
    header = ["k ↓", "", "CMC"]
    for mm in m_list:
        header.append(f"m={mm}")

    table_data = []
    for k_val in K_list:
        # from the smallest m-run
        cmc_mean_s, cmc_b_s, _, _ = results[k_val][m_small]
        # row for estimate:
        row_est = [f"{int(k_val)}", "hat{Y}", f"{cmc_mean_s:.4f}"]
        for mm in m_list:
            _, _, strat_m, _ = results[k_val][mm]
            row_est.append(f"{strat_m:.4f}")
        table_data.append(row_est)

        # row for b:
        row_b = ["", "b", f"{cmc_b_s:.4f}"]
        for mm in m_list:
            _, _, _, strat_b_m = results[k_val][mm]
            row_b.append(f"{strat_b_m:.4f}")
        table_data.append(row_b)

        # row for variance reduction ratio:
        row_ratio = ["", "Var(CMC)/Var(strat)", "1.0000"]
        for mm in m_list:
            _, cmc_b_m, _, strat_b_m = results[k_val][mm]
            ratio_val = (cmc_b_m / strat_b_m) ** 2 if strat_b_m != 0 else float('nan')
            row_ratio.append(f"{ratio_val:.4f}")
        table_data.append(row_ratio)

    print("\n--- Final Results Table ---")
    print(tabulate(table_data, headers=header, tablefmt="grid", stralign="center", numalign="center"))

    # Display all figures
    plt.show()


if __name__ == "__main__":
    main()
