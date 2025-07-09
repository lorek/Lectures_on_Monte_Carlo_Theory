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
## File: ch7_tsp_metropolis_SA_LIP_CE.py                            ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Compare several TSP algorithms using different Monte Carlo methods:
  - Cross-Entropy Tour-based method
  - Cross-Entropy Swap-based method (if enabled)
  - Cross-Entropy Update method (if enabled)
  - Metropolis with Constant Temperature (T=1)
  - Metropolis with Simulated Annealing
  - Metropolis with Locally Informed Candidate Selection
  - Metropolis-CE with Swap-based neighbors

Each algorithm is run for n_steps iterations per replication.
The experiment is replicated n_iter times. If n_iter == 1, each algorithm is run once,
its runtime and best solution are reported, and a plot is saved.
If n_iter > 1, win counts are automatically computed (based on the final result of each replication).

The final plot (from the first replication) is saved as "ch7_tsp_metropolis_SA_LIP_CE.pdf"
in the directory specified by --results_path.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from tabulate import tabulate

# --- Fixed Distance Matrix (example) ---
M = np.array([
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1744, 1693, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1836, 1200, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1693, 1836, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 1200, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0]
])
n_cities = M.shape[0]

# optimal solution (found via Held-Karp algorithm: ch7_tsp_Held-Karp_optim_solution.py )
optim_sol = 7024

# --- ParseArguments (placed immediately after M) ---
def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Compare TSP algorithms: Cross-Entropy (Tour-based), Metropolis (Constant T=1), "
                    "Metropolis (Simulated Annealing), Metropolis (Locally Informed), and Metropolis-CE (Swap-based)."
    )
    parser.add_argument('--results_path', default="results", help="Directory for saving plots (default: %(default)s)")
    parser.add_argument('--seed', default="31415", help="Seed for PCG64 RNG (default: %(default)s)")
    parser.add_argument('--n_steps', default="100", type=int, help="Number of iterations per replication (default: %(default)s)")
    parser.add_argument('--n_iter', default="1", type=int, help="Number of independent replications (default: %(default)s). If >1, win counts are computed.")
    parser.add_argument('--n_samples', default="100", type=int, help="Number of candidate samples for locally informed and CE methods (default: %(default)s)")
    parser.add_argument('--alpha', default="0.7", type=float, help="Smoothing parameter for CE method (default: %(default)s)")
    parser.add_argument('--rho', default="0.35", type=float, help="Quantile rho  for CE method (default: %(default)s)")
    parser.add_argument('--algs', default="M_const,SA, M_LIP,  M_CE, CE",
                        help="Comma-separated list of algorithms to run using short names. "
                             "Allowed values: CE (cross-entropy), M_const (Metroplis, const. temp), SA (sim. annealing),  M_LIP (Metr. locally informed proposals), M_CE (Metroplis-CE) (default: %(default)s)")
    return parser.parse_args()


# --- Utility Functions for TSP ---
def calculate_tour_distance(tour, M):
    """Return the total cyclic tour distance."""
    return sum(M[tour[i], tour[i+1]] for i in range(len(tour)-1)) + M[tour[-1], tour[0]]

def random_swap(tour, rng):
    """Return a new tour obtained by randomly swapping two cities, using rng."""
    new_tour = tour.copy()
    i, j = rng.integers(0, len(tour), size=2)
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

# --- Algorithm Implementations ---
def tsp_cross_entropy_algorithm_tour(M, n_iterations, quantile, n_samples, alpha, rng):
    n_cities = M.shape[0]
    P = np.ones((n_cities, n_cities)) / (n_cities - 1)
    np.fill_diagonal(P, 0)
    overall_best_tour = None
    overall_best_distance = float('inf')
    distances_per_iteration = []
    for iteration in range(n_iterations):
        samples = []
        distances = []
        for _ in range(n_samples):
            tour = []
            available = list(range(n_cities))
            current = rng.choice(available)
            tour.append(current)
            available.remove(current)
            while available:
                probs = P[current, available]
                probs /= probs.sum()
                next_city = rng.choice(available, p=probs)
                tour.append(next_city)
                available.remove(next_city)
                current = next_city
            samples.append(tour)
            distances.append(calculate_tour_distance(tour, M))
        best_idx = np.argmin(distances)
        best_tour = samples[best_idx]
        best_distance = distances[best_idx]
        distances_per_iteration.append(best_distance)
        if best_distance < overall_best_distance:
            overall_best_distance = best_distance
            overall_best_tour = best_tour.copy()
        # Update probability matrix P based on elite samples
        gamma = np.percentile(distances, quantile * 100)
        elite = [s for s, d in zip(samples, distances) if d <= gamma]
        new_P = np.zeros((n_cities, n_cities))
        for tour in elite:
            for i in range(n_cities - 1):
                new_P[tour[i], tour[i+1]] += 1
            new_P[tour[-1], tour[0]] += 1
        for i in range(n_cities):
            if new_P[i].sum() > 0:
                new_P[i] /= new_P[i].sum()
        P = alpha * new_P + (1 - alpha) * P
    return best_tour, overall_best_distance, distances_per_iteration, overall_best_tour, overall_best_distance

def metropolis_tsp_constant_temp(M, init_tour, n_iterations, temperature, rng):
    current_tour = init_tour.copy()
    current_distance = calculate_tour_distance(current_tour, M)
    best_tour = current_tour.copy()
    best_distance = current_distance
    overall_best_tour = best_tour.copy()
    overall_best_distance = best_distance
    distances_per_iteration = [best_distance]
    start_time = time.time()
    for _ in range(n_iterations):
        i, j = rng.integers(0, len(M), size=2)
        new_tour = current_tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_distance = calculate_tour_distance(new_tour, M)
        if new_distance < current_distance or rng.random() < np.exp((current_distance - new_distance) / temperature):
            current_tour = new_tour
            current_distance = new_distance
            if new_distance < best_distance:
                best_tour = new_tour
                best_distance = new_distance
        if best_distance < overall_best_distance:
            overall_best_tour = best_tour.copy()
            overall_best_distance = best_distance
        distances_per_iteration.append(best_distance)
    runtime = time.time() - start_time
    return best_tour, best_distance, distances_per_iteration, overall_best_tour, overall_best_distance, runtime

def metropolis_tsp_simulated_annealing(M, init_tour, n_iterations, rng):
    current_tour = init_tour.copy()
    current_distance = calculate_tour_distance(current_tour, M)
    best_tour = current_tour.copy()
    best_distance = current_distance
    overall_best_tour = best_tour.copy()
    overall_best_distance = best_distance
    distances_per_iteration = [best_distance]
    start_time = time.time()
    for k in range(1, n_iterations + 1):
        temperature =  1*np.log(k + 1)
        i, j = rng.integers(0, len(M), size=2)
        new_tour = current_tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_distance = calculate_tour_distance(new_tour, M)
        if new_distance < current_distance or rng.random() < np.exp((current_distance - new_distance) / temperature):
            current_tour = new_tour
            current_distance = new_distance
            if new_distance < best_distance:
                best_tour = new_tour
                best_distance = new_distance
        if best_distance < overall_best_distance:
            overall_best_tour = best_tour.copy()
            overall_best_distance = best_distance
        distances_per_iteration.append(best_distance)
    runtime = time.time() - start_time
    return best_tour, best_distance, distances_per_iteration, overall_best_tour, overall_best_distance, runtime

def metropolis_tsp_constant_temp_locally_informed(M, init_tour, n_iterations, n_samples, temperature, rng):
    current_tour = init_tour.copy()
    current_distance = calculate_tour_distance(current_tour, M)
    best_tour = current_tour.copy()
    best_distance = current_distance
    overall_best_tour = best_tour.copy()
    overall_best_distance = best_distance
    distances_per_iteration = [best_distance]
    start_time = time.time()
    for _ in range(n_iterations):
        samples = []
        distances = []
        for _ in range(n_samples):
            i, j = rng.integers(0, len(M), size=2)
            new_tour = current_tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            samples.append(new_tour)
            distances.append(calculate_tour_distance(new_tour, M))
        distances = np.array(distances)
        probs = np.exp(-distances / temperature)
        s = probs.sum()
        if s == 0:
            probs = np.ones(n_samples) / n_samples
        else:
            probs /= s
        idx = rng.choice(n_samples, p=probs)
        chosen_tour, chosen_distance = samples[idx], distances[idx]
        if chosen_distance < current_distance or rng.random() < np.exp((current_distance - chosen_distance) / temperature):
            current_tour = chosen_tour
            current_distance = chosen_distance
            if chosen_distance < best_distance:
                best_tour = chosen_tour
                best_distance = chosen_distance
        if best_distance < overall_best_distance:
            overall_best_tour = best_tour.copy()
            overall_best_distance = best_distance
        distances_per_iteration.append(best_distance)
    runtime = time.time() - start_time
    return best_tour, best_distance, distances_per_iteration, overall_best_tour, overall_best_distance, runtime

def metropolis_ce_tsp_swap(M, init_tour, n_iterations, n_samples, rho, alpha, rng):
    n_cities = len(init_tour)
    theta = np.full((n_cities, n_cities), 1 / (n_cities * (n_cities - 1) / 2))
    best_overall_tour = init_tour.copy()
    best_overall_distance = calculate_tour_distance(best_overall_tour, M)
    distances_per_iteration = []
    current_tour = init_tour.copy()
    current_distance = calculate_tour_distance(current_tour, M)
    start_time = time.time()
    for iteration in range(n_iterations):
        samples = []
        distances = []
        for _ in range(n_samples):
            indices = np.tril_indices(n_cities, -1)
            swap_probs = theta[indices]
            swap_probs /= swap_probs.sum()
            choice = rng.choice(len(swap_probs), p=swap_probs)
            i, j = indices[0][choice], indices[1][choice]
            new_tour = current_tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            d_new = calculate_tour_distance(new_tour, M)
            samples.append((i, j, new_tour))
            distances.append(d_new)
        distances = np.array(distances)
        gamma_t = np.percentile(distances, (1 - rho) * 100)
        elite_samples = [(i, j, tour) for (i, j, tour), d in zip(samples, distances) if d <= gamma_t]
        if elite_samples:
            counts = np.zeros((n_cities, n_cities))
            for i, j, _ in elite_samples:
                counts[i, j] += 1
                counts[j, i] += 1
            new_theta = counts / counts.sum()
            theta = alpha * new_theta + (1 - alpha) * theta
        best_idx = np.argmin(distances)
        candidate_tour = samples[best_idx][2]
        candidate_distance = distances[best_idx]
        if candidate_distance < best_overall_distance:
            best_overall_distance = candidate_distance
            best_overall_tour = candidate_tour.copy()
        current_tour = candidate_tour
        current_distance = candidate_distance
        distances_per_iteration.append(best_overall_distance)
    runtime = time.time() - start_time
    return candidate_tour, candidate_distance, distances_per_iteration, best_overall_tour, best_overall_distance, runtime

# --- Main Comparison ---
def main():
    args = ParseArguments()
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Initialize RNG using PCG64 with the provided seed.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))


    n_steps = args.n_steps
    n_iter = args.n_iter
    n_samples = args.n_samples

    # Prepare initial tour (random permutation of cities) using rng.
    init_tour = np.arange(n_cities)
    rng.shuffle(init_tour)


    # Map short names to full algorithm keys:
    alg_map = {
        "M_const": "metropolis_constant_temp",
        "SA": "metropolis_simulated_annealing",
        "M_LIP": "metropolis_constant_temp_locally_informed",
        "M_CE": "metropolis_ce_swap",
        "CE": "cross_entropy_tour",

    }

    # Default: set all algorithms to False.
    algorithm_switch = {full_name: False for full_name in alg_map.values()}
    # Enable the ones specified by the user.
    selected = [s.strip() for s in args.algs.split(',')]
    for short in selected:
        if short in alg_map:
            algorithm_switch[alg_map[short]] = True
        else:
            print(f"Warning: Unknown algorithm short name '{short}' ignored.")


    print("Algorithms to be used:")
    for alg, flag in algorithm_switch.items():
        if flag:
            print(f"  - {alg}")
    print()

    # Storage for algorithm results over replications.
    alg_results = {alg: [] for alg in algorithm_switch if algorithm_switch[alg]}
    total_runtime = {alg: 0 for alg in algorithm_switch if algorithm_switch[alg]}

    # Automatic win counting if n_iter > 1.
    if n_iter > 1:
        win_counts = {alg: 0 for alg in algorithm_switch if algorithm_switch[alg]}

    print(f"Running {n_iter} independent replications; each algorithm will run for {n_steps} steps.\n")
    rep_first = None

    for rep in range(1, n_iter + 1):
        print(f"--- Replication {rep}/{n_iter} ---")
        rep_results = {}
        # For each replication, use the same rng instance (which advances state).
        init_tour = np.arange(n_cities)
        rng.shuffle(init_tour)

        if algorithm_switch.get('metropolis_constant_temp', False):
            t0 = time.time()
            met_const = metropolis_tsp_constant_temp(M, init_tour, n_iterations=n_steps, temperature=1.0, rng=rng)
            rt = time.time() - t0
            rep_results['metropolis_constant_temp'] = (met_const[1], met_const[2])
            total_runtime['metropolis_constant_temp'] += rt
            print(f"Running Metropolis Constant T=1... \t\tFinished in {rt:.4f} sec.")
        if algorithm_switch.get('metropolis_simulated_annealing', False):
            t0 = time.time()
            met_sa = metropolis_tsp_simulated_annealing(M, init_tour, n_iterations=n_steps, rng=rng)
            rt = time.time() - t0
            rep_results['metropolis_simulated_annealing'] = (met_sa[1], met_sa[2])
            total_runtime['metropolis_simulated_annealing'] += rt
            print(f"Running Simulated Annealing... \t\t\tFinished in {rt:.4f} sec.")
        if algorithm_switch.get('metropolis_constant_temp_locally_informed', False):
            t0 = time.time()
            met_loc = metropolis_tsp_constant_temp_locally_informed(M, init_tour, n_iterations=n_steps, n_samples=n_samples, temperature=20, rng=rng)
            rt = time.time() - t0
            rep_results['metropolis_constant_temp_locally_informed'] = (met_loc[1], met_loc[2])
            total_runtime['metropolis_constant_temp_locally_informed'] += rt
            print(f"Running Metropolis Locally Informed... \t\tFinished in {rt:.4f} sec.")
        if algorithm_switch.get('cross_entropy_tour', False):
            t0 = time.time()
            ce_tour_tour, ce_distance_tour, ce_distances_per_iteration_tour, ce_overall_best_tour_tour, ce_overall_best_distance_tour = tsp_cross_entropy_algorithm_tour(
                M, n_iterations=n_steps, quantile=args.rho, n_samples=n_samples, alpha=args.alpha, rng=rng)
            rt = time.time() - t0
            rep_results['cross_entropy_tour'] = (ce_overall_best_distance_tour, ce_distances_per_iteration_tour)
            total_runtime['cross_entropy_tour'] += rt
            print(f"Running Cross-Entropy Tour... \t\t\tFinished in {rt:.4f} sec.")
        if algorithm_switch.get('metropolis_ce_swap', False):
            t0 = time.time()
            met_ce = metropolis_ce_tsp_swap(M, init_tour, n_iterations=n_steps, n_samples=n_samples, rho=0.2, alpha=0.025, rng=rng)
            rt = time.time() - t0
            rep_results['metropolis_ce_swap'] = (met_ce[4], met_ce[2])
            total_runtime['metropolis_ce_swap'] += rt
            print(f"Running Metropolis-CE Swap... \t\t\tFinished in {rt:.5f} sec.")

        # Win counting: if n_iter > 1, determine the best algorithm for this replication.
        if n_iter > 1:
            rep_final = {alg: rep_results[alg][0] for alg in rep_results}
            best_val = min(rep_final.values())
            # Determine winners by exact equality.
            winners = [alg for alg, val in rep_final.items() if val == best_val]
            fractional_win = 1.0 / len(winners)
            for alg in winners:
                win_counts[alg] += fractional_win
            print(f"WINNERS: {', '.join(winners)} (each gets {fractional_win:.3f} win)")

        for alg in rep_results:
            alg_results[alg].append(rep_results[alg])
        if rep == 1:
            rep_first = rep_results
        print()

    overall_summary = {}
    for alg, runs in alg_results.items():
        bests = [r[0] for r in runs]
        overall_summary[alg] = min(bests)

    print("\nFinal Aggregated Results:")

    if n_iter > 1:
        summary_table = []
        for alg in overall_summary:
            summary_table.append([alg, overall_summary[alg], total_runtime[alg], f"{win_counts[alg]} out of {n_iter}"])
        print(tabulate(summary_table,
                       headers=["Algorithm", "Best Distance", "Total Runtime (sec)", "Wins (out of n_iter)"],
                       tablefmt="grid"))
    else:
        summary_table = []
        for alg in overall_summary:
            summary_table.append([alg, overall_summary[alg], total_runtime[alg]])
        print(tabulate(summary_table, headers=["Algorithm", "Best Distance", "Total Runtime (sec)"], tablefmt="grid"))

    # Plot evolution of best distances from the first replication with fixed colors.
    plt.figure(figsize=(8, 4))
    if algorithm_switch.get('metropolis_constant_temp', False):
        plt.plot(range(1, len(rep_first['metropolis_constant_temp'][1]) + 1),
                 rep_first['metropolis_constant_temp'][1],
                 label="Metropolis Constant T=1", color="blue", alpha=0.5)
    if algorithm_switch.get('metropolis_simulated_annealing', False):
        plt.plot(range(1, len(rep_first['metropolis_simulated_annealing'][1]) + 1),
                 rep_first['metropolis_simulated_annealing'][1],
                 label="Simulated Annealing (T_k=1/log(k))", color="magenta", alpha=0.5)
    if algorithm_switch.get('metropolis_constant_temp_locally_informed', False):
        plt.plot(range(1, len(rep_first['metropolis_constant_temp_locally_informed'][1]) + 1),
                 rep_first['metropolis_constant_temp_locally_informed'][1],
                 label="Metropolis Locally Informed (M=100)", color="brown", alpha=0.5)
    if algorithm_switch.get('cross_entropy_tour', False):
        plt.plot(range(1, len(rep_first['cross_entropy_tour'][1]) + 1),
                 rep_first['cross_entropy_tour'][1],
                 label="Cross-Entropy", color="red", alpha=0.5, linewidth=3)
    if algorithm_switch.get('metropolis_ce_swap', False):
        plt.plot(range(1, len(rep_first['metropolis_ce_swap'][1]) + 1),
                 rep_first['metropolis_ce_swap'][1],
                 label="Metropolis-CE", color="green", alpha=0.5)

    plt.axhline(y=optim_sol, color='grey', alpha=0.7, linestyle='--', label=f'opt sol =  {optim_sol}')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_name = os.path.join(results_path, "ch7_tsp_metropolis_SA_LIP_CE.pdf")
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
    plt.xlabel("Iteration")
    plt.ylabel("Tour Distance")
    plt.title("Evolution of Best Tour Distances (First Replication)")
    print(f"\nFinal plot saved as: {fig_name}")

    plt.show()

    # print("\nFinal Results Summary:")
    # final_summary = []
    # for alg in overall_summary:
    #     final_summary.append([alg, overall_summary[alg], total_runtime[alg]])
    # print(tabulate(final_summary, headers=["Algorithm", "Best Distance", "Total Runtime (sec)"], tablefmt="grid"))

if __name__ == "__main__":
    main()
