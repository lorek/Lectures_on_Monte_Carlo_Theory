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
## File: ch1_simple_random_walk.py                                  ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import argparse
import matplotlib.pyplot as plt
from numpy.random import default_rng, PCG64
import os

def ParseArguments():
    parser = argparse.ArgumentParser(description="Simple random walk")
    parser.add_argument('--N', default="500", help='Number of points (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Path for saving results (default: %(default)s)')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()
    return args

args = ParseArguments()
N=int(args.N)           # Number of steps  (2N points)
results_path = args.results_path
os.makedirs(results_path, exist_ok=True)

# Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
seed = None if args.seed.lower() == "none" else int(args.seed)

prng = default_rng(PCG64(seed=seed))   # PCG64 with seed for reproducibility
steps = 2 * prng.integers(0, 2, size=2*N) - 1 # random steps: +1 or -1

S = np.zeros(2*N + 1)
S[1:] = np.cumsum(steps)    # Random walk with S[0]=0

# Plot the random walk
plt.figure(figsize=(8, 4))
plt.plot(S, linewidth=1, color='blue')
plt.title(f'Simple Random Walk')
plt.xlabel('Step $k$')
plt.ylabel('$S_k$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()


plt.savefig(os.path.join(results_path,"ch1_simple_random_walk.pdf"), bbox_inches='tight')

plt.show()

