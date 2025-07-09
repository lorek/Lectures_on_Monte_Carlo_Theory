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
## File: ch1_estimate_pi_quasirandom.py                             ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from numpy.random import default_rng, PCG64
from scipy.stats import qmc
from tabulate import tabulate


def ParseArguments():
    """Parse command-line arguments for the simulation."""
    parser = argparse.ArgumentParser(description="Estimating Pi with quasirandom numbers")
    parser.add_argument('--R', default="1024", help='Total number of points (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Path for saving results (default: %(default)s)')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()
    return args


def main():

    # Parse arguments
    args = ParseArguments()
    R = int(args.R)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    ###########
    ## PLOTS ##
    ###########

    # Initialize PRNG (PCG64) and generate pseudorandom points
    rng = default_rng(PCG64(seed=31415))
    U1 = rng.random(R)
    U2 = rng.random(R)

    # Plot Pseudorandom points
    plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, 1, 100), np.sqrt(1 - np.linspace(0, 1, 100)**2), color='black', alpha=0.4)
    plt.scatter(U1, U2, s=2, color='black')
    plt.savefig(os.path.join(results_path, "ch1_estimate_pi_pcg64.pdf"), bbox_inches='tight')
    plt.title("Pseudorandom (PCG64)")


    # Lattice (Scipy QMC)
    lh = qmc.LatinHypercube(d=2, seed=31415)
    lh_points= lh.random(R)

    plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, 1, 100), np.sqrt(1 - np.linspace(0, 1, 100)**2), color='black', alpha=0.4)
    plt.scatter(lh_points[:, 0], lh_points[:, 1], s=2, color='black')
    plt.savefig(os.path.join(results_path, "ch1_estimate_pi_latin_hypercube.pdf"), bbox_inches='tight')
    plt.title("LatinHypercube")

    # Sobol (Scipy QMC)
    sobol = qmc.Sobol(d=2, scramble=True, seed=31415)
    sobol_points = sobol.random(R)

    plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, 1, 100), np.sqrt(1 - np.linspace(0, 1, 100)**2), color='black', alpha=0.4)
    plt.scatter(sobol_points[:, 0], sobol_points[:, 1], s=2, color='black')
    plt.savefig(os.path.join(results_path, "ch1_estimate_pi_sobol.pdf"), bbox_inches='tight')
    plt.title("Sobol")


    # Halton (Scipy QMC)
    halton = qmc.Halton(d=2, scramble=True, seed=31415)
    halton_points = halton.random(R)

    plt.figure(figsize=(7, 7))
    plt.plot(np.linspace(0, 1, 100), np.sqrt(1 - np.linspace(0, 1, 100)**2), color='black', alpha=0.4)
    plt.scatter(halton_points[:, 0], halton_points[:, 1], s=2, color='black')
    plt.savefig(os.path.join(results_path, "ch1_estimate_pi_halton.pdf"), bbox_inches='tight')
    plt.title("Halton")


    #################
    ## ESTIMATIONS ##
    #################

    # R values for estimation
    R_values = [2**8, 2**9, 2**10, 2**11]

    # Collect results
    results = []

    for R in R_values:
        # --- PCG64 (Pseudorandom) ---
        rng = default_rng(PCG64(seed=31415))
        U1 = rng.random(R)
        U2 = rng.random(R)
        Y_pcg = np.mean((U1**2 + U2**2 <= 1) * 4)
        error_pcg = np.abs(Y_pcg - np.pi)

        # --- Lattice (QMC) ---
        lattice = qmc.LatinHypercube(d=2, seed=31415)
        lh_points= lattice.random(R)
        Y_lattice = np.mean((lh_points[:, 0]**2 + lh_points[:, 1]**2 <= 1) * 4)
        error_lattice = np.abs(Y_lattice - np.pi)

        # --- Sobol (QMC) ---
        sobol = qmc.Sobol(d=2, scramble=True, seed=31415)
        sobol_points = sobol.random(R)
        Y_sobol = np.mean((sobol_points[:, 0]**2 + sobol_points[:, 1]**2 <= 1) * 4)
        error_sobol = np.abs(Y_sobol - np.pi)

        # --- Halton (QMC) ---
        halton = qmc.Halton(d=2, scramble=True, seed=31415)
        halton_points = halton.random(R)
        Y_halton = np.mean((halton_points[:, 0]**2 + halton_points[:, 1]**2 <= 1) * 4)
        error_halton = np.abs(Y_halton - np.pi)

        # Store results in the same style as LaTeX
        results.append([f"2^{int(np.log2(R))}", "Estimate",
                        f"{Y_pcg:.5f}", f"{Y_lattice:.5f}", f"{Y_sobol:.5f}", f"{Y_halton:.5f}"])
        results.append(["", "Error",
                        f"{error_pcg:.5f}", f"{error_lattice:.5f}", f"{error_sobol:.5f}", f"{error_halton:.5f}"])

    # Display results using tabulate
    print(tabulate(results, headers=["R", "", "PCG64", "Lattice", "Sobol", "Halton"], tablefmt="grid"))


    plt.show()

if __name__ == '__main__':
    main()

