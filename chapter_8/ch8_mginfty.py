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
## File: ch8_mginfty.py                                             ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.stats import pareto
from tabulate import tabulate

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Simulate M/M/∞ and multiple M/G/∞(Pareto) queues, show final L(t_max) in a table."
    )
    parser.add_argument("--results_path", default="results",
                        help="Directory to save plots (default: %(default)s)")
    parser.add_argument("--seed", default="31415",
                        help="Seed for PCG64 RNG (use 'none' for no fixed seed) (default: %(default)s)")
    parser.add_argument("--t_max", type=float, default=10000.0,
                        help="Maximum simulation time (default: %(default)s)")

    # M/M/∞ parameters
    parser.add_argument("--lambda_rate", type=float, default=5.0,
                        help="Arrival rate lambda for M/M/∞ (default: %(default)s)")
    parser.add_argument("--mu_rate", type=float, default=1.0,
                        help="Service rate mu for M/M/∞ (default: %(default)s)")

    # M/G/∞(Pareto) parameters
    parser.add_argument("--pareto_alphas", default="3.2,2.2,1.2",
                        help="Comma-separated shape alphas for M/G/∞ (Pareto) (default: %(default)s)")
    parser.add_argument("--pareto_factors", default="2.2,1.2,0.2",
                        help="Comma-separated scale factors for M/G/∞ (Pareto) (default: %(default)s)")

    return parser.parse_args()


def simulate_mm_infty(rng, t_max, lam, mu):
    """
    Simulate M/M/∞ queue up to time t_max:
      arrivals ~ Exp(lam)
      service ~ Exp(mu)
    For t=1..t_max (integer), L(t) = [∑ (min(depart_i,t) - min(arrive_i,t)) ] / t.
    Returns a list (index= t-1 => L(t)).
    """
    arrival_time = 0.0
    arrival_times = []
    departure_times = []

    # Generate arrivals and departures
    while arrival_time < t_max:
        delta = rng.exponential(1.0 / lam)
        arrival_time += delta
        if arrival_time > t_max:
            break
        arrival_times.append(arrival_time)
        # service
        serv = rng.exponential(1.0 / mu)
        departure_times.append(arrival_time + serv)

    # For each t in [1..t_max]
    results = []
    n = len(arrival_times)
    for t in range(1, int(t_max) + 1):
        sum_t = 0.0
        i = 0
        while i < n and arrival_times[i] < t:
            start_t = min(arrival_times[i], t)
            end_t   = min(departure_times[i], t)
            if end_t > start_t:
                sum_t += (end_t - start_t)
            i += 1
        results.append(sum_t / t)
    return results


def simulate_mg_infty_pareto(rng, t_max, lam, alpha, factor):
    """
    M/G/∞ with Pareto service: T_service = factor * ((U^(-1/alpha)) -1).
    arrivals ~ Exp(lam)
    For t=1..t_max, L(t) by the same approach.
    """
    arrival_time= 0.0
    arrival_times= []
    departure_times= []

    while arrival_time< t_max:
        delta = rng.exponential(1.0/lam)
        arrival_time+= delta
        if arrival_time> t_max:
            break
        arrival_times.append(arrival_time)

        U= rng.random()
        service = factor * ( (U**(-1.0/alpha)) -1.0 )
        departure_times.append(arrival_time+ service)

    # integer-based results
    results=[]
    n= len(arrival_times)
    for t in range(1, int(t_max)+1):
        sum_t= 0.0
        i=0
        while i<n and arrival_times[i]< t:
            st= min(arrival_times[i], t)
            et= min(departure_times[i], t)
            if et> st:
                sum_t+=(et-st)
            i+=1
        results.append(sum_t/t)
    return results


def main():
    args = ParseArguments()
    os.makedirs(args.results_path, exist_ok=True)

    # Setup RNG
    seed_str = args.seed.lower()
    if seed_str == "none":
        rng = np.random.default_rng()
        print("Using a random seed (none).")
    else:
        seed_val = int(seed_str)
        rng = np.random.default_rng(seed_val)
        print(f"Using seed={seed_val}")

    t_max = args.t_max
    lam   = args.lambda_rate
    mu    = args.mu_rate

    # parse the lists for M/G/∞
    alpha_list  = [s.strip() for s in args.pareto_alphas.split(',') if s.strip()]
    factor_list = [s.strip() for s in args.pareto_factors.split(',') if s.strip()]
    if len(alpha_list)!= len(factor_list):
        print("ERROR: mismatch length of --pareto_alphas and --pareto_factors.")
        return

    # We'll store final results in a table
    final_results = []


    # 1) M/M/∞
    print(f"\nSimulating M/M/∞: lam={lam}, mu={mu}, t_max={t_max} ...")
    mm_results = simulate_mm_infty(rng, t_max, lam, mu)
    mm_final = mm_results[-1]  # L(t_max)
    scenario_name= f"M/M/∞(λ={lam}, μ={mu})"
    final_results.append([scenario_name, f"{mm_final:.4f}"])

    # Plot
    outmm= os.path.join(args.results_path, "ch8_mginfty_mminfty.pdf")
    xvals= range(1, int(t_max)+1)
    plt.figure(figsize=(5,3))
    plt.ylim([0.5, 6])
    plt.yticks([1, 2, 3, 4, 5])
    plt.plot(xvals, mm_results, label="Time-dependent average L(t)")
    true_l_mm1 =  lam/mu
    plt.axhline(y=true_l_mm1, color='red', alpha=0.7, linestyle='--', label=f'True l')

    plt.grid(True)
    plt.savefig(outmm, bbox_inches='tight')
    plt.title(scenario_name)
    plt.xlabel("time t")
    plt.ylabel("L(t)")
    plt.legend()
    print(f"Saved {outmm}")

    # 2) M/G/∞(Pareto)
    for (a_str, f_str) in zip(alpha_list, factor_list):
        alpha_val= float(a_str)
        factor_val= float(f_str)
        if alpha_val > 1.0:
            eS = factor_val / (alpha_val - 1.0)
        else:
            eS = float('inf')

        true_l_mm1 = lam / eS #
        print(f"\nSimulating M/G/∞ with S = {factor_val}*X, X ~ Pareto(α={alpha_val}), E[S]={eS:.4f}, t_max={t_max}...")


        scen= f"S={factor_val}*Pareto(α={alpha_val})"
        #print(f"\nSimulating M/G/∞ with {scen}, t_max={t_max} ...")
        mg_results= simulate_mg_infty_pareto(rng, t_max, lam, alpha_val, factor_val)
        mg_final= mg_results[-1]
        final_results.append([scen, f"{mg_final:.4f}"])

        # Plot
        outmg= os.path.join(args.results_path,
                            f"ch8_mginfty_Par_alpha{alpha_val}_factor{factor_val}.pdf")
        xvals2= range(1, int(t_max)+1)
        plt.figure(figsize=(5,3))
        plt.ylim([0.5, 6])
        plt.yticks([1, 2, 3, 4, 5])
        plt.plot(xvals2, mg_results, label="Time-dependent average L(t)")
        plt.grid(True)


        plt.axhline(y=true_l_mm1, color='red', alpha=0.7, linestyle='--', label=f'True l')

        plt.savefig(outmg, bbox_inches='tight')
        plt.title(f"M/G/∞ {scen} (λ={lam})")
        plt.xlabel("time t")
        plt.ylabel("L(t)")
        plt.legend()
        print(f"Saved {outmg}")

    # Tabulate final results
    print("\nFinal results at t_max:\n")
    print(tabulate(final_results, headers=["Scenario", "L(t_max)"], tablefmt="github"))

    plt.show()


if __name__ == "__main__":
    main()
