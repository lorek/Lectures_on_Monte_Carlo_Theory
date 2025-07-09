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
## File: ch4_pi_required_R_absolute_relative.py                     ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import norm
import os

def ParseArguments():
    parser = argparse.ArgumentParser(description="Plot three functions in the style of a Maple example.")
    parser.add_argument('--results_path', default="results",
                        help="Folder to save the output plot (default: %(default)s)")
    parser.add_argument('--delta', default="0.01", help="max R (default: %(default)s)")


    args = parser.parse_args()
    return args

def main():
    args = ParseArguments()
    delta =  float(args.delta)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)



    I = np.pi



    # Create a range of R values from 1 to 30 (n_grid points)
    epsilon_grid = np.linspace(0.2, 1.0, 100)




    z_val = norm.ppf(1-delta/2)
    print("z_(1-",delta,"/2) = ", z_val)

    VarY1 = 16*np.pi/4*(1-np.pi/4)
    f_abs = z_val**2*VarY1/(epsilon_grid**2)

    f1 = 3 * np.log(2 / delta) / (epsilon_grid ** 2 * I)
    f2 = 3 * np.log(2 / delta) / ((epsilon_grid/I) ** 2  * I)


    # Plot them
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epsilon_grid, f_abs, linestyle='-', color='red',  label="$R^{abs}$")
    ax.plot(epsilon_grid, f1, linestyle='--', color='green', label="$R^{rel,1}$")
    ax.plot(epsilon_grid, f2, linestyle=':', color='blue', label="$R^{rel,2}$")

    ax.legend()

    # Save figure
    outpath = os.path.join(results_path, "ch4_pi_required_R_absolute_relative.pdf")
    plt.savefig(outpath, bbox_inches='tight',pad_inches=0)
    print(f"Plot saved to {outpath}")

    ax.set_xlabel("ε")
    ax.set_ylabel("R")
    ax.set_title("Nr of replications $R$ to make bounds for π at most δ="+str(delta))

    plt.show()

if __name__ == "__main__":
    main()
