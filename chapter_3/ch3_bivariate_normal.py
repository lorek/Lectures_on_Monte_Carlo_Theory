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
## File: ch3_bivariate_normal.py                                    ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from numpy.random import default_rng, PCG64


def sample_bivariate_normal(rng, n, var1, var2, rho):
    cov = rho * np.sqrt(var1 * var2)
    mean = np.array([0.0, 0.0])
    cov_mat = np.array([[var1, cov], [cov, var2]])
    samples = rng.multivariate_normal(mean, cov_mat, size=n)
    return samples[:, 0], samples[:, 1]


def ParseArguments():
    parser = argparse.ArgumentParser(description="Generate bivariate normal scatter plots.")
    parser.add_argument('--results_path', default="results", help="Output folder for plots")
    parser.add_argument('--n_biv', default="500", help="Number of points per plot")
    parser.add_argument('--seed', default="31415", help="Random seed")
    return parser.parse_args()


def main():
    args = ParseArguments()
    n_biv = int(args.n_biv)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed))

    configs = [
        (1.0, 1.0, 0.0, "std_rho0"),
        (3.0, 1.0, 0.6, "var3_rho0.6"),
        (3.0, 1.0, 0.9, "var3_rho0.9"),
    ]

    for var1, var2, rho, tag in configs:
        X1, X2 = sample_bivariate_normal(rng, n_biv, var1, var2, rho)
        plt.figure(figsize=(4, 4))
        plt.scatter(X1, X2, color='blue', alpha=0.8, s=8)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.savefig(os.path.join(results_path, f"ch3_biv_normal_{tag}.pdf"),
                    bbox_inches='tight', pad_inches=0)
        plt.title(f"Bivariate Normal: var1={var1}, var2={var2}, rho={rho}")

    print(f"Bivariate plots saved in folder: {results_path}")
    plt.show()


if __name__ == "__main__":
    main()
