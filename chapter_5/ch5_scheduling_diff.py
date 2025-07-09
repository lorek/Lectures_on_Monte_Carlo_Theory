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
## File: ch5_scheduling_diff.py                                     ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import argparse
from tabulate import tabulate
from numpy.random import default_rng, PCG64

def ParseArguments():
    """
    Parse command-line arguments:
      --N : number of tasks (default 10)
      --R : comma-separated list of replication counts
            default "1000,100000"
    """
    parser = argparse.ArgumentParser(description='Estimate (C^SRPT - C^LRPT) with CRN vs. separate draws.')
    parser.add_argument('--N', type=int, default=10,
                        help='Number of tasks to schedule, default: %(default)s')
    parser.add_argument('--R', type=str, default='1000,100000',
                        help='Comma-separated list of replication counts, e.g., "1000,100000". Default: %(default)s')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")
    args = parser.parse_args()

    return args



def makespan_SRPT(X):
    """
    Given an array X of Exp(1) processing times, compute the makespan
    under SRPT (Shortest Remaining Processing Time) with 2 servers.
    """
    Y = np.sort(X)
    m = len(Y)
    total_time = 0.0

    while m > 1:
        t_add = min(Y[0], Y[1])
        total_time += t_add
        Y[0] = abs(Y[1] - Y[0])
        Y = np.delete(Y, 1)
        m -= 1
        Y = np.sort(Y)

    total_time += Y[0]
    return total_time


def makespan_LRPT(X):
    """
    Given an array X of Exp(1) processing times, compute the makespan
    under LRPT (Longest Remaining Processing Time) with 2 servers.
    """
    Z = np.sort(X)
    m = len(Z)
    total_time = 0.0

    while m > 1:
        t_add = min(Z[-1], Z[-2])
        total_time += t_add
        Z[-2] = abs(Z[-1] - Z[-2])
        Z = Z[:-1]
        m -= 1
        Z = np.sort(Z)

    total_time += Z[0]
    return total_time


def simulate_diff_crn(R, N,rng):
    """
    CRN version: For each replication:
      - Generate ONE array of N i.i.d. Exp(1) times
      - Compute SRPT makespan and LRPT makespan with that same array
      - Take the difference
    """
    diffs = []
    for _ in range(R):
        X = -np.log(rng.uniform(0,1,N))  # single set of random times
        srpt_time = makespan_SRPT(X)
        lrpt_time = makespan_LRPT(X)
        diffs.append(srpt_time - lrpt_time)

    diffs = np.array(diffs)
    mean_val = diffs.mean()
    std_val = diffs.std(ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf


def simulate_diff_separate(R, N, rng):
    """
    Separate-random‐draws version:
      - Generate array X for SRPT, array W for LRPT (independent draws)
      - Compute difference
    """
    diffs = []
    for _ in range(R):
        X = -np.log(rng.uniform(0,1,N))  # times for SRPT
        W = -np.log(rng.uniform(0,1,N))  # independent times for LRPT
        srpt_time = makespan_SRPT(X)
        lrpt_time = makespan_LRPT(W)
        diffs.append(srpt_time - lrpt_time)

    diffs = np.array(diffs)
    mean_val = diffs.mean()
    std_val = diffs.std(ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf


def main():
    args = ParseArguments()
    N = int(args.N)
    Rs = args.R
    # Convert the comma-separated list of R values into integers
    R_list = [int(r.strip()) for r in Rs.split(',')]

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))


    for R in R_list:
        print(f"\n=== Estimating (C^SRPT - C^LRPT) with N={N} tasks, R={R} replications ===")
        mean_crn, std_crn, half_crn = simulate_diff_crn(R, N,rng)
        mean_sep, std_sep, half_sep = simulate_diff_separate(R, N, rng)

        # Build data table
        table_data = [
            ["CRN", f"{mean_crn:.4f}", f"{std_crn:.4f}", f"{half_crn:.4f}"],
            ["CMC/Separate", f"{mean_sep:.4f}", f"{std_sep:.4f}", f"{half_sep:.4f}"]
        ]
        # Print pretty table
        print(tabulate(
            table_data,
            headers=["Method", "Mean (I^)", "Std (S^)", "Half-CI (b^)"],
            tablefmt="pretty"
        ))


if __name__ == "__main__":
    main()