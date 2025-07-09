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
## File: ch7_knapsack_dynamic_prog_optim_solution.py                ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import time
import numpy as np


def main():
    # Problem setup
    d = 100
    W_max = 3000
    weights = np.arange(1, d + 1)  # [1,2,...,100]
    values = np.array([i ** 1.2 for i in range(1, d + 1)])

    # Start timing
    t0 = time.time()

    # We'll use a 1D DP approach where dp[w] = best possible value with capacity w
    dp = np.zeros(W_max + 1, dtype=float)  # dp array from 0..W_max

    for i in range(d):
        w_i = weights[i]
        v_i = values[i]
        # Traverse capacity backward to avoid reusing item i multiple times
        for w in range(W_max, w_i - 1, -1):
            # If we can include this item, see if it improves the best known value
            candidate_value = dp[w - w_i] + v_i
            if candidate_value > dp[w]:
                dp[w] = candidate_value

    best_value = dp[W_max]

    elapsed = time.time() - t0

    print(f"Exact DP solution for d={d}, W_max={W_max}:")
    print(f"  Best total value = {best_value:.4f}")
    print(f"  Running time = {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
