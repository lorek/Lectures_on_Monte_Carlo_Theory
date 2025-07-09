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
## File: ch7_tsp_Held-Karp_optim_solution.py                        ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Solve the TSP optimally for 13 cities using the Held–Karp algorithm (Dynamic Programming).
This algorithm runs in O(n^2 * 2^n) time, which is efficient for n = 13,
in contrast to brute-force which runs in O(n!)
It assumes the tour starts at city 0.
"""

import numpy as np
import time

# Distance matrix for 13 cities
M = np.array([
    [0,    2451, 713,  1018, 1631, 1374, 2408, 213,  2571, 875,  1420, 2145, 1972],
    [2451, 0,    1745, 1524, 831,  1240, 959,  2596, 403,  1589, 1374, 357,  579],
    [713,  1745, 0,    355,  920,  803,  1737, 851,  1858, 262,  940,  1453, 1260],
    [1018, 1524, 355,  0,    700,  862,  1395, 1123, 1584, 466,  1056, 1280, 987],
    [1631, 831,  920,  700,  0,    663,  1021, 1769, 949,  796,  879,  586,  371],
    [1374, 1240, 803,  862,  663,  0,    1681, 1551, 1765, 547,  225,  887,  999],
    [2408, 959,  1737, 1395, 1021, 1681, 0,    2493, 678,  1744, 1693, 1114, 701],
    [213,  2596, 851,  1123, 1769, 1551, 2493, 0,    2699, 1038, 1836, 1200, 2099],
    [2571, 403,  1858, 1584, 949,  1765, 678,  2699, 0,    1744, 1645, 653,  600],
    [875,  1589, 262,  466,  796,  547,  1724, 1038, 1744, 0,    679,  1272, 1162],
    [1420, 1374, 940,  1056, 879,  225,  1693, 1836, 1645, 679,  0,    1017, 1200],
    [2145, 357,  1453, 1280, 586,  887,  1114, 1200, 653,  1272, 1017, 0,    504],
    [1972, 579,  1260, 987,  371,  999,  701,  2099, 600,  1162, 1200, 504,  0]
])
n = M.shape[0]  # n = 13

def held_karp_tsp(M):
    n = M.shape[0]
    dp = { (1, 0): 0 }  # Base case: only city 0 is visited.

    # Iterate over subsets of size s = 2, 3, ..., n that include city 0.
    for s in range(2, n + 1):
        for mask in [m for m in range(1, 1 << n) if bin(m).count("1") == s and (m & 1)]:
            for j in range(n):
                if mask & (1 << j) and j != 0:
                    prev_mask = mask ^ (1 << j)
                    if s == 2:
                        # Only possible predecessor is city 0.
                        dp[(mask, j)] = dp[(prev_mask, 0)] + M[0, j]
                    else:
                        dp[(mask, j)] = min(
                            dp[(prev_mask, k)] + M[k, j]
                            for k in range(n) if (prev_mask & (1 << k)) and k != 0
                        )
    full_mask = (1 << n) - 1
    optimal_cost = min(dp[(full_mask, j)] + M[j, 0] for j in range(1, n))
    return optimal_cost




if __name__ == '__main__':
    start_time = time.time()
    optimal_cost = held_karp_tsp(M)
    end_time = time.time()
    print("Optimal TSP tour cost (Held–Karp):", optimal_cost, ", found in ", np.round(end_time - start_time,5), "sec.")

