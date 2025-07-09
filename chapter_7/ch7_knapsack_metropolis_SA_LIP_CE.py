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
## File: ch7_knapsack_metropolis_SA_LIP_CE.py                       ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Usage Example:
  python refactored_knapsack.py --algorithm metropolis,LIP_full, CE_all_coord,metropolis_ce \
                                --n_steps 100 --n_samples 20 --alpha 0.001 --seed 31415
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from tabulate import tabulate


# -------------------------------------------------------------------------
#  1) DEFINE PROBLEM INSTANCE
# -------------------------------------------------------------------------
d = 100
W_max = 3000
weights = np.arange(1, d + 1)  # 1,2,3,...,100
values = np.array([i ** 1.2 for i in range(1, d + 1)])

# optimal solution for this example (found by ch7_knapsack_dynamic_prog_optim_solution.py )
optimal_value = 7251.6602

# -------------------------------------------------------------------------
#  2) PARSE ARGUMENTS
# -------------------------------------------------------------------------
def ParseArguments():
    parser = argparse.ArgumentParser(description="Refactored knapsack algorithms with old color scheme.")
    parser.add_argument('--results_path', default="results", help="Directory for saving plots (default: %(default)s)")
    parser.add_argument('--seed', default="31415", help="Seed for PCG64 RNG (default: %(default)s)")
    parser.add_argument('--n_steps', default=100, type=int, help="Number of iterations per replication (default: %(default)s)")
    parser.add_argument('--n_iter', default=1, type=int, help="Number of independent replications (default: %(default)s)")
    parser.add_argument('--n_samples', default=30, type=int, help="Number of local candidates or CE sample size (default: %(default)s)")
    parser.add_argument('--rho', default="0.6", type=float, help="Quantile rho  for CE methods (default: %(default)s)")
    parser.add_argument('--alpha_ce', default=0.7, type=float,
                        help="Smoothing parameter for CE method (default: %(default)s)")
    parser.add_argument('--alpha_metropolis_ce', default=0.001, type=float,
                    help="Smoothing parameter for Metropolis-CE method (default: %(default)s)")
    parser.add_argument('--algs', default="Metropolis,LIP,LIP_full,CE,Metropolis-CE",
                        help="Comma‐separated list of algorithms to run. "
                             "Allowed values:  Metropolis, CE (cross-entropy), LIP (locally informed proposals with n_samples), LIP_full (LIP with full), Metropolis-CE (default: %(default)s)")

    #"Allowed values: CE (cross-entropy), M_const (Metroplis, const. temp), SA (sim. annealing),  M_LIP (Metr. locally informed proposals), M_CE (Metroplis-CE) (default: %(default)s)")


    return parser.parse_args()


# -------------------------------------------------------------------------
#  3) UTILITY FUNCTIONS
# -------------------------------------------------------------------------
def knapsack_value(solution, values):
    return np.dot(solution, values)

def knapsack_weight(solution, weights):
    return np.dot(solution, weights)


# -------------------------------------------------------------------------
#  4) ALGORITHMS (mirroring your old code)
# -------------------------------------------------------------------------

def knapsack_metropolis_algorithm(weights, values, W_max, T, iterations, rng):
    """
    Standard Metropolis for 0-1 Knapsack, e.g. T=5 in old code.
    """
    d = len(weights)
    X = np.zeros(d, dtype=int)
    cur_w = knapsack_weight(X, weights)
    cur_v = knapsack_value(X, values)

    w_hist = [cur_w]
    v_hist = [cur_v]

    best_sol = X.copy()
    best_val = cur_v
    best_w = cur_w

    for _ in range(iterations):
        i = rng.integers(0, d)
        X_new = X.copy()
        X_new[i] = 1 - X_new[i]

        new_w = knapsack_weight(X_new, weights)
        new_v = knapsack_value(X_new, values)

        if new_w <= W_max:
            delta = new_v - cur_v
            if delta > 0 or rng.random() < np.exp(delta / T):
                X = X_new
                cur_w = new_w
                cur_v = new_v
                if cur_v > best_val:
                    best_sol = X.copy()
                    best_val = cur_v
                    best_w = cur_w

        w_hist.append(cur_w)
        v_hist.append(cur_v)

    return w_hist, v_hist, best_sol, best_w, best_val


def knapsack_metropolis_locally_informed(weights, values, W_max, T, iterations, n_samples, rng):
    """
    LIP version (randomly propose n_samples single‐bit flips, pick with some weighting).
    Typically T=100, n_samples=100, etc.
    """


    d = len(weights)
    X = np.zeros(d, dtype=int)
    cur_w = knapsack_weight(X, weights)
    cur_v = knapsack_value(X, values)

    w_hist = [cur_w]
    v_hist = [cur_v]

    best_sol = X.copy()
    best_val = cur_v
    best_w = cur_w



    for _ in range(iterations):
        candidates = []
        c_w = []
        c_v = []
        for __ in range(n_samples):
            X_new = X.copy()
            i = rng.integers(0, d)
            X_new[i] = 1 - X_new[i]
            nw = knapsack_weight(X_new, weights)
            nv = knapsack_value(X_new, values)
            if nw <= W_max:
                candidates.append(X_new)
                c_w.append(nw)
                c_v.append(nv)

        if len(candidates) == 0:
            w_hist.append(cur_w)
            v_hist.append(cur_v)
            continue

        c_v = np.array(c_v)
        # You might use e.g. exp(beta * c_v / T) or something. We'll do exp(-c_v / T):
        exp_vals = np.exp(c_v / T)
        s = exp_vals.sum()
        if s == 0:
            probs = np.ones(len(candidates)) / len(candidates)
        else:
            probs = exp_vals / s

        idx = rng.choice(len(candidates), p=probs)
        X = candidates[idx]
        cur_w = c_w[idx]
        cur_v = c_v[idx]

        if cur_v > best_val:
            best_sol = X.copy()
            best_val = cur_v
            best_w = cur_w

        w_hist.append(cur_w)
        v_hist.append(cur_v)

    return w_hist, v_hist, best_sol, best_w, best_val


def knapsack_metropolis_locally_informed_full(weights, values, W_max, T, iterations, rng):
    """
    LIP variant with n_samples = d (a 'full' approach).
    """
    return knapsack_metropolis_locally_informed(weights, values, W_max, T, iterations, n_samples=d, rng=rng)



def knapsack_cross_entropy(weights, values, W_max, iterations, n_samples, alpha, rng,rho):
    """
    CE method that uses a probability vector for each item being included,
    picks 'best' or 'elite' solutions, and updates that vector with smoothing alpha.
    """

    d = len(weights)
    theta = np.full(d, 0.5)

    best_sol = np.zeros(d, dtype=int)
    best_val = 0
    best_w = 0

    w_hist = []
    v_hist = []

    for _ in range(iterations):
        samples = []
        perf = []
        for __ in range(n_samples):
            cand = (rng.random(d) < theta).astype(int)
            w_c = np.dot(cand, weights)
            if w_c <= W_max:
                v_c = np.dot(cand, values)
                samples.append(cand)
                perf.append(v_c)

        if len(perf) == 0:
            # No valid solutions
            w_hist.append(best_w)
            v_hist.append(best_val)
            continue

        arr_perf = np.array(perf)

        # (1-rho) quantile

        gamma_t = np.percentile(arr_perf, (1 - rho) * 100)



        elite = [samples[i] for i, pval in enumerate(arr_perf) if pval >= gamma_t]

        if len(elite) > 0:
            new_theta = np.mean(elite, axis=0)
            # Smooth update
            theta = alpha * new_theta + (1 - alpha) * theta
            #theta = np.clip(theta, 0, 1)

            # Optional: if you want a floor or bounding like [0.1, 0.9], do
            # new_theta = 0.1 + 0.8 * np.mean(elite, axis=0)
            # etc.

        # Update global best
        best_iter_val = arr_perf.max()
        if best_iter_val > best_val:
            best_val = best_iter_val
            idx_best = np.argmax(arr_perf)
            best_sol = samples[idx_best].copy()
            best_w = np.dot(best_sol, weights)

        w_hist.append(best_w)
        v_hist.append(best_val)

    return w_hist, v_hist, best_sol, best_w, best_val



def knapsack_metropolis_ce(weights, values, W_max, iterations, n_samples, alpha, rng, rho):
    """
    Example 'Metropolis-CE' hybrid.
    We keep a distribution over bits, sample single flips, pick best, update distribution.
    """
    d = len(weights)
    theta = np.full(d, 1.0 / d)

    X = np.zeros(d, dtype=int)
    cur_v = knapsack_value(X, values)
    cur_w = knapsack_weight(X, weights)



    best_sol = X.copy()
    best_val = cur_v
    best_w = cur_w

    w_hist = [cur_w]
    v_hist = [cur_v]

    for _ in range(iterations):
        candidates = []
        perf = []
        for __ in range(n_samples):
            idx = rng.choice(d, p=theta)
            X_new = X.copy()
            X_new[idx] = 1 - X_new[idx]
            nw = knapsack_weight(X_new, weights)
            nv = knapsack_value(X_new, values)
            if nw <= W_max:
                candidates.append(X_new)
                perf.append(nv)

        if len(perf) == 0:
            w_hist.append(best_w)
            v_hist.append(best_val)
            continue

        arr_perf = np.array(perf)
        idx_best = np.argmax(arr_perf)
        local_best_val = arr_perf[idx_best]
        local_best_sol = candidates[idx_best]

        # update global best
        if local_best_val > best_val:
            best_val = local_best_val
            best_sol = local_best_sol.copy()
            best_w = knapsack_weight(best_sol, weights)

        gamma_t = np.percentile(arr_perf, (1 - rho) * 100)


        elite_indices = np.where(arr_perf >= gamma_t)[0]
        elite_solutions = [candidates[i] for i in elite_indices]

        #print(">>>>>>> rho = ", rho, " gamma_t = ", gamma_t, " elite_indices = ",elite_indices)

        # ---- 3) Frequencies of bits = fraction of elite solutions that have item j=1
        if len(elite_solutions) > 0:
            freq = np.mean(elite_solutions, axis=0)  # shape = (d,)
        else:
            # no elites (should be rare, but if it happens, skip update)
            freq = np.zeros(d)

        #  "bounded" freq -> map to [0.1, 0.9]
        #mapped_freq = 0.1 + 0.8 * freq
        mapped_freq =  freq

        # ---- 4) Smooth update of theta
        new_theta = mapped_freq
        #print("NEW THETA = ", theta[:6])

#        alpha=0.0005
        #print(">>>>>>>>>>>>>>>>>>> alpha = ",alpha, " , n_samples = ", n_samples, " , rho = ",rho)
        theta = alpha * new_theta + (1 - alpha) * theta


        #
        #
        # update distribution
        # e.g. push probabilities near the chosen local best
        #new_theta = np.where(local_best_sol == 1, 0.9, 0.1)
        #theta = alpha * new_theta + (1 - alpha) * theta

        # Clip or ensure in [1e-5, 1-1e-5] if needed
        theta = np.clip(theta, 1e-5, 1 - 1e-5)


        # normalize so sum=1 if you interpret it that way:
        sum_theta = theta.sum()
        if sum_theta > 0:
            theta /= sum_theta


        # adopt that local best for next iteration
        X = local_best_sol
        cur_v = local_best_val
        cur_w = knapsack_weight(X, weights)

        w_hist.append(best_w)
        v_hist.append(best_val)

    return w_hist, v_hist, best_sol, best_w, best_val

# -------------------------------------------------------------------------
#  5) MAIN COMPARISON
# -------------------------------------------------------------------------
def main():
    args = ParseArguments()
    os.makedirs(args.results_path, exist_ok=True)

    #  -- Initialize RNG --
    seed_str = args.seed.lower()
    if seed_str == "none":
        seed = None
        print("No fixed seed. Using random seed.")
    else:
        seed = int(seed_str)
        print(f"Using RNG seed = {seed}")
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    # Collect user parameters
    n_steps = args.n_steps
    n_iter = args.n_iter
    rho = args.rho
    n_samples = args.n_samples # in book: M, but here M i the distance matrix
    alpha_ce = args.alpha_ce
    alpha_metropolis_ce = args.alpha_metropolis_ce
    alg_list = [x.strip() for x in args.algs.split(',') if x.strip()]

    print("Algorithms requested:", alg_list)
    print(f"n_steps={n_steps}, n_iter={n_iter}, n_samples={n_samples}, alpha_ce={alpha_ce} , alpha_metropolis_ce={alpha_metropolis_ce}")

    # We store results: { alg_name : [ (list_of_weights, list_of_values), ... ] } for each rep
    results_all = {a: [] for a in alg_list}
    total_runtime = {a: 0.0 for a in alg_list}

    # If n_iter>1, track wins
    if n_iter > 1:
        win_counts = {a: 0.0 for a in alg_list}

    print(f"Running {n_iter} independent replications; each algorithm will run for {n_steps} steps.\n")


    for rep in range(1, n_iter + 1):
        print(f"\n--- Replication {rep}/{n_iter} ---")
        rep_results = {}

        for alg in alg_list:
            t0 = time.time()
            if alg == "Metropolis":
                w_hist, v_hist, best_sol, best_w, best_val = knapsack_metropolis_algorithm(
                    weights, values, W_max, T=5, iterations=n_steps, rng=rng)
            elif alg == "LIP":
               # print(">>>>>>> n_samples = ",n_samples)
                w_hist, v_hist, best_sol, best_w, best_val = knapsack_metropolis_locally_informed(
                    weights, values, W_max, T=100, iterations=n_steps, n_samples=n_samples, rng=rng)
            elif alg == "LIP_full":
                w_hist, v_hist, best_sol, best_w, best_val = knapsack_metropolis_locally_informed_full(
                    weights, values, W_max, T=100, iterations=n_steps, rng=rng)
            elif alg == "CE":
                w_hist, v_hist, best_sol, best_w, best_val = knapsack_cross_entropy(
                    weights, values, W_max, iterations=n_steps, n_samples=n_samples, alpha=alpha_ce, rng=rng,rho=rho)
            elif alg == "Metropolis-CE":
                w_hist, v_hist, best_sol, best_w, best_val = knapsack_metropolis_ce(
                    weights, values, W_max, iterations=n_steps, n_samples=n_samples, alpha=alpha_metropolis_ce, rng=rng, rho=rho)
            else:
                print(f"  [Warning] Unknown algorithm: {alg}")
                continue

            rt = time.time() - t0
            total_runtime[alg] += rt
            rep_results[alg] = (w_hist, v_hist, best_val)
            print(f"Running {alg} ... \t\t\tFinished in {rt:.4f} sec. Best value = {best_val:.3f}.")
            #print(f"  -> {alg} done in {rt:.2f}s. Best value = {best_val:.3f}.")

        if n_iter > 1 and rep_results:
            # Win counting
            bestvals = {a: rep_results[a][2] for a in rep_results}
            the_best = max(bestvals.values())
            winners = [a for a, val in bestvals.items() if val == the_best]
            share = 1.0 / len(winners)
            for w in winners:
                win_counts[w] += share
            print(f"   Winners: {winners}, each +{share}")

        for alg in rep_results:
            results_all[alg].append(rep_results[alg])

    # ---------------------------------------------------------------------
    #   SUMMARIZE
    # ---------------------------------------------------------------------
    print("\nSummary:")
    if n_iter == 1:
        rows = []
        for alg in alg_list:
            if len(results_all[alg]) == 0:
                continue
            # single replication -> results_all[alg][0] = (w_hist, v_hist, best_val)
            best_val = results_all[alg][0][2]
            rows.append([alg, best_val, f"{total_runtime[alg]:.2f}"])
        print(tabulate(rows, headers=["Algorithm", "Best Value", "Time (s)"], tablefmt="grid"))
    else:
        rows = []
        for alg in alg_list:
            if len(results_all[alg]) == 0:
                continue
            # gather best values from each rep
            best_vals = [res[2] for res in results_all[alg]]
            overall_best = max(best_vals) if best_vals else 0
            rows.append([alg, overall_best, f"{total_runtime[alg]:.2f}", f"{win_counts[alg]:.2f}/{n_iter}"])
        print(tabulate(rows, headers=["Algorithm", "Best of All Reps", "Total Time", "Wins"], tablefmt="grid"))

    # ---------------------------------------------------------------------
    #   PLOT: two subplots, reusing the old color scheme
    # ---------------------------------------------------------------------
    # We'll just show the first replication's traces.
    rep_plot = 0  # index 0 => 1st replication
    plt.figure(figsize=(8, 4))

    color_map = {
        "Metropolis": "blue",
        "LIP": "brown",
        "LIP_full": "orange",
        "CE": "red",
        "Metropolis-CE": "green",
    }


    # 2) Right subplot: Values
    #plt.subplot(1, 2, 2)
    #plt.figure()
    for alg in alg_list:
        if len(results_all[alg]) == 0:
            continue
        w_hist, v_hist, best_val = results_all[alg][rep_plot]
        c = color_map.get(alg, "black")
        plt.plot(v_hist, label=f"Total value: {alg}", color=c, alpha=0.7)

    plt.axhline(y=optimal_value, color='grey', alpha=0.7, linestyle='--', label=f'Optimal value =  {optimal_value}')


    plt.legend()
    plt.grid(True)


    fig_name = os.path.join(args.results_path, "ch7_knapsack_metropolis_SA_LIP_CE.pdf")

    plt.tight_layout()
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    print(f"\nPlot saved to {fig_name}")

    plt.title("Knapsack Value vs Iteration ")
    plt.xlabel("Iteration")
    plt.ylabel("Knapsack Value")



    # Optional:
    plt.show()
    # End


if __name__ == "__main__":
    main()
