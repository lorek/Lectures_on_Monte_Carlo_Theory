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
## File: ch8_aloha.py                                               ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Slotted ALOHA (classic): each of X_n packets transmits with prob h. Exactly 1 attempt = success."
    )
    parser.add_argument("--results_path", default="results",
                        help="Directory to save plots (default: %(default)s)")
    parser.add_argument("--t_max", type=int, default=10000,
                        help="Number of slots to simulate (default: %(default)s)")
    parser.add_argument("--lambda_rate", type=float, default=0.31,
                        help="Poisson arrival rate per slot (default: %(default)s)")
    parser.add_argument("--h", type=float, default=0.1,
                        help="Transmission probability for each packet in backlog (default: %(default)s)")
    parser.add_argument("--seed", default="31415",
                        help="Seed for PCG64 RNG (use 'none' for no fixed seed) (default: %(default)s)")
    return parser.parse_args()


def main():
    args = ParseArguments()

    # Create results directory
    os.makedirs(args.results_path, exist_ok=True)

    # Initialize RNG with PCG64
    seed_str = args.seed.lower()
    if seed_str == "none":
        rng = np.random.default_rng(np.random.PCG64())
        print("Using a random seed.")
    else:
        seed = int(seed_str)
        rng = np.random.default_rng(np.random.PCG64(seed=seed))
        print(f"Using seed = {seed}")

    t_max = args.t_max
    lambda_rate = args.lambda_rate
    h = args.h

    print(f"Slotted ALOHA (classic) with t_max={t_max}, lambda={lambda_rate}, h={h}")

    # Arrays to store data over time
    X_vals = np.zeros(t_max+1, dtype=int)   # backlog size X_n
    gamma_vals = np.zeros(t_max+1, dtype=float)  # local throughput gamma_n

    successes = 0
    # initial condition
    X_vals[0] = 0

    for n in range(t_max):
        Xn = X_vals[n]

        # arrivals ~ Poisson(lambda_rate)
        arrivals = rng.poisson(lambda_rate)
        Xtemp = Xn + arrivals

        # Each of Xtemp packets transmits with prob h => binomial(Xtemp, h)
        # If exactly 1 transmits => success
        attempts = rng.binomial(Xtemp, h)

        if attempts == 1:
            # success
            X_next = Xtemp - 1
            successes += 1
        else:
            # either 0 or >=2 attempts => no success
            X_next = Xtemp

        X_vals[n+1] = X_next
        gamma_vals[n+1] = successes / (n+1)

    # --- Plot #1: X_n over time
    #plt.figure(figsize=(7.5,4))
    plt.figure(figsize=(5, 5))
    plt.plot(range(t_max+1), X_vals, label="Number of packages",linewidth = 1)
    plt.grid(True)
    out1 = os.path.join(args.results_path, f"ch8_aloha_nr_packages.pdf")
    plt.savefig(out1, bbox_inches="tight")
    plt.xlabel("Slot n")
    plt.ylabel("Backlog X_n")
    plt.title("Slotted ALOHA: backlog size X_n (classic model)")
    plt.legend()
    print(f"Saved figure {out1}")


    # --- Plot #2: local throughput gamma_n
    plt.figure(figsize=(5,5))
    plt.plot(range(t_max+1), gamma_vals, label="Throughput")
    plt.grid(True)
    out2 = os.path.join(args.results_path, f"ch8_aloha_throughput.pdf")
    plt.savefig(out2, bbox_inches="tight")
    plt.xlabel("Slot n")
    plt.ylabel("Local Throughput gamma_n")
    plt.title("Slotted ALOHA: local throughput gamma_n")
    plt.legend()
    print(f"Saved figure {out2}")

    plt.show()


if __name__ == "__main__":
    main()
