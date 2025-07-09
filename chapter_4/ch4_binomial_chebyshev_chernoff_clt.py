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
## File: ch4_binomial_chebyshev_chernoff_clt.py                     ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def ParseArguments():
    parser = argparse.ArgumentParser(description="Plot three functions in the style of a Maple example.")
    parser.add_argument('--results_path', default="results",
                        help="Folder to save the output plot (default: %(default)s)")
    parser.add_argument('--R_min', default="1", help="max R (default: %(default)s)")
    parser.add_argument('--R_max', default="40", help="max R (default: %(default)s)")
    parser.add_argument('--n_grid', default="300", help="max R (default: %(default)s)")
    args = parser.parse_args()
    return args

def main():
    args = ParseArguments()
    n_grid = int(args.n_grid)
    R_min = int(args.R_min)
    R_max = int(args.R_max)
    results_path = args.results_path


    os.makedirs(results_path, exist_ok=True)

    # Create a range of R values from 1 to 30 (n_grid points)
    R = np.linspace(R_min,  R_max, n_grid)

    # Define the three functions
    f1 = 4.0 / R
    f2 = 2.0 * np.exp(-R / 24.0)
    f3 = np.sqrt(8.0 / (np.pi * R)) * np.exp(-R / 8.0)

    # Plot them
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(R, f1, linestyle='-', color='blue',  label=r"$4/R$")
    ax.plot(R, f2, linestyle='--', color='green', label=r"$2 exp(-R/24)$")
    ax.plot(R, f3, linestyle=':', color='red',   label=r"$\sqrt{8/(\pi R)}exp(-R/8)$")

    ax.legend()

    # Save figure
    outpath = os.path.join(results_path, "ch4_binomial_chebyshev_chernoff_clt.pdf")
    plt.savefig(outpath, bbox_inches='tight',pad_inches=0)
    print(f"Plot saved to {outpath}")

    ax.set_xlabel("R")
    ax.set_ylabel("Function Value")
    ax.set_title("Comparison of Chebyshev, Chernoff and CLT bounds")

    plt.show()

if __name__ == "__main__":
    main()
