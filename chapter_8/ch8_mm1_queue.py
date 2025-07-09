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
## File: ch8_mm1_queue.py                                           ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from fractions import Fraction

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Simulate an M/M/1 queue for multiple values of lambda, saving separate plots."
    )
    parser.add_argument("--results_path", default="results",
                        help="Directory to save plots (default: %(default)s)")
    parser.add_argument("--seed", default="31415",
                        help="Seed for PCG64 RNG (use 'none' for no fixed seed) (default: %(default)s)")
    parser.add_argument("--t_max", default="10000",
                        help="Number of steps to simulate  (use 'none' for no fixed seed) (default: %(default)s)")
    parser.add_argument("--mu_rate", type=float, default=1.0,
                        help="Service rate mu (default: %(default)s)")
    parser.add_argument("--lambda_rate", default="0.5,8/9,1,1.2",
                        help="Comma-separated list of arrival rates lambda (default: %(default)s)")

    return parser.parse_args()


def simulate_mm1(lambda_rate, mu_rate, t_max, rng):
    """
    Run a single simulation of an M/M/1 queue up to time t_max,
    given arrival rate = lambda_rate, service rate = mu_rate.
    Returns (time, state_list, l_hat), where:
      - time: array of time points (length n+1)
      - state_list: number of tasks in the system at each time index
      - l_hat: time-average number of tasks
    """
    state = 0  # initial state
    time = [0.0]
    state_list = [state]

    while time[-1] < t_max:
        # If state=0, only arrivals happen => rate = lambda;
        # else arrivals + departures => rate = lambda + mu
        if state == 0:
            rate = lambda_rate
        else:
            rate = lambda_rate + mu_rate

        # Draw next event time from exponential(1/rate)
        time_to_next = rng.exponential(1.0 / rate)
        next_time = time[-1] + time_to_next

        # If the next event would go beyond t_max, break
        if next_time > t_max:
            break

        time.append(next_time)

        # Determine whether arrival or departure
        # Probability of arrival = lambda_rate / rate
        if rng.random() < (lambda_rate / rate):
            state += 1
        else:
            state = max(0, state - 1)

        state_list.append(state)

    # Add a final point at t_max
    final_time_interval = t_max - time[-1]
    time.append(t_max)
    state_list.append(state)

    # Compute time-average number of tasks
    time_intervals = np.diff(time)  # durations in each state
    # We pair each interval with state_list[:-1].
    l_hat = np.sum(np.array(state_list[:-1]) * time_intervals) / t_max

    return time, state_list, l_hat


def main():
    args = ParseArguments()

    # Create results directory if needed
    os.makedirs(args.results_path, exist_ok=True)

    # Initialize RNG using PCG64 with the provided seed.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))
    print(f"Using seed = {seed}")

    # Parse mu rate
    mu_rate = args.mu_rate

    # Parse list of lambda rates
    lambdas_str = args.lambda_rate.split(',')
    lambda_rates = []
    for lam_str in lambdas_str:
        lam_str = lam_str.strip()
        if lam_str:
            if "/" in lam_str:
                val = float(Fraction(lam_str))  # interpret "8/9" as a fraction
            else:
                val = float(lam_str)
            lambda_rates.append(val)

    t_max = int(args.t_max)

    print(f"mu_rate = {mu_rate}, lambda_rates = {lambda_rates}, t_max={t_max}")

    # Loop over each lambda, run the simulation, produce a separate plot
    for lam_val in lambda_rates:
        time, state_list, l_hat = simulate_mm1(lam_val, mu_rate, t_max, rng)

        # Print results
        print("\n--- Simulation with lambda =", lam_val, "---")
        rho = lam_val / mu_rate if mu_rate != 0 else float('inf')
        print(f"Parameters: mu={mu_rate}, lambda={lam_val}, rho={rho:.4f}")
        print(f"Estimated long-term average # of tasks l_hat = {l_hat:.6f}")

        if mu_rate > lam_val:
            # Theoretical formula if rho < 1
            l_true = lam_val / (mu_rate - lam_val)
            rel_err = (l_hat / l_true - 1)
            print(f"Exact l = {l_true:.6f}")
            abs_rel_err = abs(rel_err)
            print(f"Relative error = {abs_rel_err:.6f}, i.e., {abs_rel_err:.6%}")
        else:
            print("System is unstable (rho >= 1), no finite theoretical mean.")

        # Plot
        plt.figure(figsize=(7.5,4.5))
        plt.step(time, state_list, where='post')

        plt.title(" ")
        plt.grid(True)

        # Construct filename
        # E.g. ch8_mm1_queue_mu1.0_lambda0.5.pdf

        mu_str = f"{mu_rate:.2f}".replace('.', '_')
        lam_str = f"{lam_val:.2f}".replace('.', '_')

        out_filename = f"ch8_mm1_queue_mu{mu_str}_lambda{lam_str}.pdf"
        out_path = os.path.join(args.results_path, out_filename)
        plt.savefig(out_path, bbox_inches='tight')
        plt.title(f"M/M/1 Queue (lambda={lam_val}, mu={mu_rate})")

        print(f"Plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
