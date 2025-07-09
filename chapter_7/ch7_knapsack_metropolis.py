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
## File: ch7_knapsack_metropolis.py                                 ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Simulate a Metropolis-like algorithm for a knapsack problem.
Here, the knapsack is represented as a binary vector x of dimension d (each element indicates
whether an item is included). Each of the d items has a "weight" and a "value" (computed as
cumulative sums raised to specified powers). At each iteration, a random bit of x is flipped,
and the change is accepted only if the new total weight (x · weights) is below the threshold W,
and with a probability that depends on the total value (x · values) of the knapsack.
The evolution of the dot products x·weights and x·values is recorded and then plotted.
The final figure is saved in a specified results directory.
"""

import argparse
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import os
from numpy.random import default_rng, PCG64

def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Simulate a Metropolis-like algorithm for a knapsack problem using a binary vector."
    )
    parser.add_argument('--d', default="100", type=int,
                        help="Number of items (dimension of knapsack, default: 100)")
    parser.add_argument('--R', default="150", type=int,
                        help="Number of iterations (default: 200)")
    parser.add_argument('--T', default="1", type=float,
                        help="Temperature parameter (default: 1)")
    parser.add_argument('--W', default="3000", type=float,
                        help="Maximum allowed total weight (default: 3000)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG; use 'None' for no fixed seed (default: %(default)s)")
    parser.add_argument('--results_path', default="results",
                        help="Directory for saving plots (default: results)")
    return parser.parse_args()


def simulate_metropolis(d, R, T, W, rng):
    """
    Simulate the Metropolis-like update for the knapsack problem.

    The knapsack is represented as a binary vector x of dimension d. Each item i (1 <= i <= d)
    has a weight and a value. Here, weights and values are defined as:

      weights = cumsum(ones(d))^1.0
      values  = cumsum(ones(d))^1.2

    This means that the i-th item has weight proportional to i and value proportional to i^(1.2).
    The total weight is computed as np.dot(x, weights) and must remain below W.
    The acceptance probability is computed using the current total value np.dot(x, values)
    and a temperature parameter T.

    Parameters:
      d   : Number of items (dimension of x).
      R   : Number of iterations.
      T   : Temperature parameter.
      W   : Maximum allowed total weight.
      rng : Random number generator.

    Returns:
      x             : Final binary vector (knapsack solution).
      xweights_all  : Array of total weights (x · weights) over iterations.
      xvalues_all   : Array of total values (x · values) over iterations.
      weights, values: The weight and value vectors.
    """
    # Define weight and value vectors (item i has weight proportional to i and value proportional to i^(1.2))
    weights = np.cumsum(np.ones(d)) ** 1.0
    values = np.cumsum(np.ones(d)) ** 1.2

    # Initialize knapsack x as a binary vector of zeros (item not chosen).
    x = np.zeros(d, dtype=int)

    # Arrays to record evolution of total weight and total value.
    xweights_all = np.zeros(R)
    xvalues_all = np.zeros(R)

    for k in range(R):
        xweights_all[k] = np.dot(x, weights)
        xvalues_all[k] = np.dot(x, values)

        # Randomly choose an item and propose to flip its inclusion.
        i = rng.integers(low=0, high=d)
        x2 = cp.copy(x)
        x2[i] = 1 - x2[i]

        # Accept the proposed change only if the new total weight is below the threshold W.
        if np.dot(x2, weights) < W:
            U = rng.random()
            # Compute acceptance probability based on the current total value.
            prob = 1 / (1 + np.exp(- (1 / T) * np.dot(values, x)))
            if U < prob:
                x[i] = 1
            else:
                x[i] = 0

    return x, xweights_all, xvalues_all, weights, values


def plot_results(R, xweights_all, xvalues_all, W, results_path):
    """
    Plot the evolution of the knapsack's total weight and value over iterations.
    Saves the figure as "ch7_knapsack_metropolis_evolution.pdf" in the results_path directory.
    """
    steps = np.arange(R)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, xweights_all, color='blue', label="Total weight (x · weights)")
    ax.plot(steps, xvalues_all, color='red', label="Total value (x · values)")
    ax.plot(steps, W * np.ones(R), color='gray', alpha=0.6, label="Max weight W")

    ax.legend()


    os.makedirs(results_path, exist_ok=True)
    save_path = os.path.join(results_path, "ch7_knapsack_metropolis_evolution.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("Total weight / value")
    plt.tight_layout()
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    args = ParseArguments()

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.strip().lower() == "none" else int(args.seed)
    rng =  default_rng(PCG64(seed=seed))

 
    # Simulate the Metropolis updates for the knapsack problem.
    x, xweights_all, xvalues_all, weights, values = simulate_metropolis(args.d, args.R, args.T, args.W, rng)

    print("Final knapsack (binary vector):", x)
    print("Final total weight (x · weights):\t", np.round(np.dot(x, weights),1))
    print("Final total value (x · values):\t\t", np.round(np.dot(x, values),2))

    plot_results(args.R, xweights_all, xvalues_all, args.W, args.results_path)


if __name__ == "__main__":
    main()
