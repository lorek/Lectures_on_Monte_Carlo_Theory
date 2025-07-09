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
## File: ch5_strat_sampling_pi.py                                   ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##


import numpy as np
from numpy.random import default_rng, PCG64
import matplotlib.pyplot as plt
import argparse
import os
from tabulate import tabulate

def parse_arguments():
    parser = argparse.ArgumentParser(description="Stratified Sampling for Estimating π")
    parser.add_argument('--R', default="200,10000",
                        help="Comma-separated list of total replications (default: %(default)s)")
    parser.add_argument('--Rpilot', default="100",
                        help="Number of pilot replications (default: %(default)s)")
    parser.add_argument('--m', default="5,10,20",
                        help="Comma-separated list of number of strata (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Folder for saving outputs (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()
    return args


def simulate_stratum(m, allocation, rng):
    """
    For each stratum j=1..m with allocation[j-1] replications:
      Sample U ~ U(0,1), set V = (j-1)/m + U/m,
      then Y = 4 * sqrt(1 - V^2).
    Returns concatenated arrays: Y, stratum indices, U, and V.
    """
    Y_all = np.array([])
    stratum_idx = np.array([], dtype=int)
    U_all = np.array([])
    V_all = np.array([])
    for j in range(1, m + 1):
        nj = allocation[j - 1]
        if nj < 1:
            nj = 1
        U = rng.uniform(0,1,nj)
        V = (j - 1) / m + U / m
        Y = 4.0 * np.sqrt(1.0 - V ** 2)
        Y_all = np.concatenate((Y_all, Y))
        stratum_idx = np.concatenate((stratum_idx, j * np.ones(nj, dtype=int)))
        U_all = np.concatenate((U_all, U))
        V_all = np.concatenate((V_all, V))
    return Y_all, stratum_idx, U_all, V_all


def pilot_simulation(m, Rpilot, rng):
    """
    Pilot simulation: equally allocate Rpilot among m strata.
    Returns:
      pilot_means, pilot_stds (each an array of length m),
      pilot_alloc (the integer allocation used).
    """
    w_uniform = np.ones(m) / m
    pilot_alloc = (w_uniform * Rpilot).astype(int)
    pilot_alloc[pilot_alloc == 0] = 1
    pilot_alloc[-1] = Rpilot - np.sum(pilot_alloc[:-1])

    Y_all, strata_idx, _, _ = simulate_stratum(m, pilot_alloc, rng)
    pilot_means = np.zeros(m)
    pilot_stds = np.zeros(m)
    for j in range(1, m + 1):
        vals = Y_all[strata_idx == j]
        pilot_means[j - 1] = np.mean(vals)
        pilot_stds[j - 1] = np.std(vals, ddof=1)
    return pilot_means, pilot_stds, pilot_alloc


def optimal_allocation(m, pilot_stds, R_total):
    """
    Optimal allocation: allocate R_total replications among m strata in proportion to pilot_stds.
    Ensures each stratum gets at least 1 replication and that the sum equals R_total.
    """
    w = pilot_stds / np.sum(pilot_stds)
    alloc = (w * R_total).astype(int)
    alloc[alloc == 0] = 1
    alloc[-1] = R_total - np.sum(alloc[:-1])
    return alloc


def stratified_estimator(m, allocation,rng):
    """
    Run stratified sampling with the given allocation for m strata.
    Returns: (est, var_est, Y_all, stratum_idx, U_all, V_all).
    """
    Y_all, stratum_idx, U_all, V_all = simulate_stratum(m, allocation,rng)
    p_j = np.ones(m) / m
    means_j = np.zeros(m)
    vars_j = np.zeros(m)
    for j in range(1, m + 1):
        vals = Y_all[stratum_idx == j]
        means_j[j - 1] = np.mean(vals)
        if len(vals) > 1:
            vars_j[j - 1] = np.var(vals, ddof=1) / len(vals)
        else:
            vars_j[j - 1] = 0.0

    est = np.sum(p_j * means_j)
    var_est = np.sum((p_j ** 2) * vars_j)
    return est, var_est, Y_all, stratum_idx, U_all, V_all


def cmc_estimator(R, rng):
    """
    Classical Monte Carlo estimator for π using Y = 4*sqrt(1-U^2).
    """
    U = rng.uniform(0,1,R)
    Y = 4 * np.sqrt(1 - U ** 2)
    est = np.mean(Y)
    var_est = np.var(Y, ddof=1) / R
    std_est = np.sqrt(var_est)
    #return est, 1.96 * np.sqrt(var_est), abs(est - np.pi)
    return est, std_est, U, Y



######################################
#            Main Code               #
######################################

def main():
    args = parse_arguments()
    # Convert comma-separated strings into lists of integers.
    R_list = [int(x.strip()) for x in args.R.split(",")]
    m_list = [int(x.strip()) for x in args.m.split(",")]
    R_pilot = int(args.Rpilot)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    # Dictionary to store optimal allocations for each (m, R) pair from pilot simulation.
    optimal_allocations = {}
    pilot_std_dict = {}
    for m in m_list:
        _, pilot_stds, pilot_alloc = pilot_simulation(m, R_pilot, rng)
        pilot_std_dict[m] = pilot_stds  # store pilot stds for m
        for R_val in R_list:
            optimal_allocations[(m, R_val)] = optimal_allocation(m, pilot_stds, R_val)

    ###########################################
    # 1) Pilot Simulation Table (Table V.1)   #
    ###########################################
    # For each m, produce a table with columns: "Stratum 1", ..., "Stratum m"
    # Rows: first row: pilot standard deviations (S'_j), then one row per R value (optimal allocation R_j)
    for m in m_list:
        pilot_stds = pilot_std_dict[m]
        header = ["Stratum j = "] + [str(j) for j in range(1, m + 1)]
        table_data = []
        # First row: Pilot standard deviations
        row_std = ["Pilot Std"] + [f"{pilot_stds[j - 1]:.3f}" for j in range(1, m + 1)]
        table_data.append(row_std)
        # One row per R value (optimal allocation)
        for R_val in R_list:
            alloc_opt = optimal_allocations[(m, R_val)]
            row_alloc = [f"R = {R_val}, R_j = "] + [str(alloc_opt[j - 1]) for j in range(1, m + 1)]
            table_data.append(row_alloc)
        print("\n=== Optimal allocation from Rpilot = ", R_pilot," simulations for m = ",m," ===")
        print(tabulate(table_data, headers=header, tablefmt="grid"))

    ###########################################
    # 2) Full Stratified Simulation Results   #
    #     (separate table for each R)         #
    ###########################################
    # For each R, produce a table with rows: "$\hat{Y}_R$", "$\widehat{Var}(Y_R)$", "Error"
    # and columns: "CMC", "Stratified m=5", "Stratified m=10", "Stratified m=20" (for each m in m_list)

    cmc_results = {} # needed later for plot
    tables_by_R = {}
    for R_val in R_list:
        # Compute CMC estimator
        cmc_est, cmc_std, U_cmc, Y_cmc  = cmc_estimator(R_val, rng)
        cmc_results[R_val] = {
            "cmc_est": cmc_est,
            "cmc_std": cmc_std,
            "U": U_cmc,
            "Y": Y_cmc
        }
        cmc_abs = np.abs(np.pi-cmc_est)
        headers = ["Metric", "CMC"] + [f"Stratified m={m_val}" for m_val in m_list]
        row_est = [r"hat{Y}_R", f"{cmc_est:.5f}"]
        row_std = [r"sqrt(\widehat{Var}(Y_R))", f"{(cmc_std) :.5f}"]
        row_err_b = [r"b=1.96*sqrt(\widehat{Var}(Y_R))", f"{(1.96*cmc_std) :.5f}"]
        row_err_abs = ["Absolute error |π-hat{Y}_R|", f"{cmc_abs:.5f}"]
        for m_val in m_list:
            alloc_opt = optimal_allocations[(m_val, R_val)]
            est_strat, var_strat, _, _, _, _ = stratified_estimator(m_val, alloc_opt,rng)
            row_est.append(f"{est_strat:.5f}")
            row_std.append(f"{np.sqrt(var_strat):.5f}")
            row_err_b.append(f"{1.96*np.sqrt(var_strat):.5f}")
            row_err_abs.append(f"{abs(est_strat - np.pi):.5f}")
        table = tabulate([row_est,   row_std, row_err_b, row_err_abs], headers=headers, tablefmt="grid")
        tables_by_R[R_val] = table
    for R_val in sorted(tables_by_R.keys()):
        print(f"\n--- CMC and  Stratified Simulation Results  for R = {R_val} ---")
        print(tables_by_R[R_val])

    ###########################################
    # 3) Final Plots                          #
    ###########################################
    # Use the smallest R in R_list and m=10 if available; otherwise, use the first m.
    R_final = min(R_list)
    m_final = 10 if 10 in m_list else m_list[0]


    ## Stratified (optimal alloction) plot
    alloc_plot = optimal_allocations[(m_final, R_final)]
    est_plot, var_plot, Y_all_plot, strata_idx_plot, U_plot, V_plot = stratified_estimator(m_final, alloc_plot,rng)

    xx = np.linspace(0, 1, 300)
    yy = 4 * np.sqrt(1 - xx ** 2)


    fig, ax = plt.subplots(figsize=(6, 4.5))
    # Plot the underlying curve (the function 4*sqrt(1-V^2))
    ax.plot(xx, yy, color='gray', alpha=0.5)
    # Scatter the stratified sample points
    ax.scatter(V_plot, Y_all_plot, s=2, color='black')

    # Fill vertical stripes for each stratum and annotate with the optimal allocation.
    for i in range(1, m_final + 1):
        xs = (i - 1) / m_final
        xn = i / m_final
        # Alternate the fill transparency for visual clarity.
        if i % 2 == 1:
            ax.fill([xs, xn, xn, xs], [0, 0, 4.2, 4.2], color='gray', alpha=0.2)
        else:
            ax.fill([xs, xn, xn, xs], [0, 0, 4.2, 4.2], color='gray', alpha=0.3)
    # Place text in each stratum indicating the optimal allocation R_j.
    for i in range(m_final):
        ax.text(i / m_final + 1 / (2 * m_final), 0.15, str(alloc_plot[i]),
                horizontalalignment='center', verticalalignment='center')


    plt.tight_layout()
    outfig = os.path.join(results_path, "ch5_strat_sampling_pi_str.pdf")

    plt.savefig(outfig, bbox_inches='tight', pad_inches=0)
    ax.set_xlabel("V")
    ax.set_ylabel(r"$4\sqrt{1-V^2}$")
    ax.set_title(f"Stratified Sampling: R = {R_final}, m = {m_final}")
    print(f"\nFigure saved to: {outfig}")



    ## CMC and resulting (random) alloctation
    cmc_stored_results = cmc_results[R_final]
    U_cmc = cmc_stored_results["U"]
    Y_cmc = cmc_stored_results["Y"]

    bins_cmc = np.linspace(0, 1, m_final + 1)
    counts_cmc, _ = np.histogram(U_cmc, bins=bins_cmc)

    # Prepare the underlying curve for Y = 4*sqrt(1-U^2)
    xx = np.linspace(0, 1, 300)
    yy = 4 * np.sqrt(1 - xx ** 2)

    fig_cmc, ax_cmc =   plt.subplots(figsize=(6, 4.5))
    ax_cmc.plot(xx, yy, color='gray', alpha=0.5, label=r"$4\sqrt{1-U^2}$")
    ax_cmc.scatter(U_cmc, Y_cmc, s=2, color='black')
    # Fill vertical stripes for each stratum and annotate with the optimal allocation.
    for i in range(1, m_final + 1):
        xs = (i - 1) / m_final
        xn = i / m_final
        # Alternate the fill transparency for visual clarity.
        if i % 2 == 1:
            ax_cmc.fill([xs, xn, xn, xs], [0, 0, 4.2, 4.2], color='gray', alpha=0.2)
        else:
            ax_cmc.fill([xs, xn, xn, xs], [0, 0, 4.2, 4.2], color='gray', alpha=0.3)
    # Place text in each stratum indicating the optimal allocation R_j.
    for i in range(m_final):
        ax_cmc.text(i / m_final + 1 / (2 * m_final), 0.15, str(counts_cmc[i]),
                horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    outfig_cmc = os.path.join(results_path, "ch5_strat_sampling_pi_cmc.pdf")
    plt.savefig(outfig_cmc, bbox_inches='tight', pad_inches=0)
    ax_cmc.set_xlabel("U")
    ax_cmc.set_ylabel(r"$4\sqrt{1-U^2}$")
    ax_cmc.set_title(f"CMC Estimator: R = {R_final}")
    print(f"\nCMC figure saved to: {outfig_cmc}")
    plt.show()


if __name__ == "__main__":
    main()
