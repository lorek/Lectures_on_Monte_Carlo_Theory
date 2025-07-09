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
## File: ch5_ContrVariates_Cauchy.py                                ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from math import pi

def parse_arguments():
    """
    Parse command-line arguments. For instance:
    python script.py --results_path my_results
    """
    parser = argparse.ArgumentParser(description="Compute 4th-degree polynomial approximation for k_Y(x) and plot with k2_X(x).")
    parser.add_argument('--results_path', default="results",
                        help='Path for saving results (default: %(default)s)')
    args = parser.parse_args()
    return args


def k_Y(x):
    """k_Y(x) = 2 / (pi * (1 + x^2))."""
    return 2.0 / (np.pi * (1.0 + x**2))

def k1(x):
    """k^{(1)}_X(x) = 2/pi - 4/(5*pi)*x."""
    return 2.0/np.pi - (4.0/(5.0*np.pi))*x

def main():
    args = parse_arguments()
    results_path = args.results_path

    # Ensure the output folder exists
    os.makedirs(results_path, exist_ok=True)

    # 1) Points at which we match k_Y(x)
    x_vals = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    y_vals = k_Y(x_vals)  # k_Y at these points

    # 2) Build the Vandermonde system for a 4th-degree polynomial
    #    f(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    #    We solve V * [a0..a4]^T = y_vals
    V = np.vander(x_vals, 5, increasing=True)  # shape is (5,5)

    coeffs = np.linalg.solve(V, y_vals)        # [a0, a1, a2, a3, a4]

    # 3) Generate a fine grid for plotting
    x_grid = np.linspace(0, 2, 201)
    # Evaluate k_Y(x) on the grid
    y_ky = k_Y(x_grid)
    # Evaluate the polynomial f(x) on the grid (np.polyval wants descending order)
    # Coeffs are [a0, a1, a2, a3, a4], so reverse them:
    y_k4 = np.polyval(coeffs[::-1], x_grid)
    # Evaluate k2_X(x) on the grid
    y_k1= k1(x_grid)

    # 4) Plot all three
    plt.figure(figsize=(8, 4))

    plt.style.use("default")  # ensures a clean default style
    plt.rcParams["axes.grid"] = False
    ax = plt.gca()
    ax.grid(False)

    # Remove all spines
    ax = plt.gca()
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    # (Optional) set plot limits so you can see the arrows
    # e.g., if your x in [0,2] and y in [0,1.2], extend a bit:
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.02, 0.7)

    # Draw the x-axis as a line at y=0
    ax.axhline(0, color='black', linewidth=1)
    # Add an arrow at the right end of the x-axis
    ax.plot(ax.get_xlim()[1], 0, ">k", clip_on=False, markersize=6)

    # Draw the y-axis as a line at x=0
    ax.axvline(0, color='black', linewidth=1)
    # Add an arrow at the top end of the y-axis
    ax.plot(0, ax.get_ylim()[1], "^k", clip_on=False, markersize=6)

    plt.plot(x_grid, y_ky, label=r"$k_Y(v)$", color="red")
    plt.plot(x_grid, y_k4, label=r"$k^{(4)}_X(v) $", color="blue", linestyle="--")
    plt.plot(x_grid, y_k1, label=r"$k^{(1)}_X(v) $", color="green", linestyle=":")

    plt.legend()


    # 5) Save the figure
    out_file = os.path.join(results_path, "ch5_Cauchy_CV.pdf")
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Figure saved to: {out_file}")
    plt.xlabel("x")
    plt.ylabel("function value")
    plt.title("k_Y(v) and fitted k1(v) (1st deg poly) and k4(v) (4th-deg pol)")

    # 6) Print coefficients
    print("Polynomial coefficients (f(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4):")
    for i, c in enumerate(coeffs):
        print(f"  a{i} = {c:g}")

    # Optional check: show f(x) vs k_Y(x) at the five original points
    print("\nCheck at x in {0, 0.5, 1, 1.5, 2}:")
    for xv, yv in zip(x_vals, y_vals):
        f_val = np.polyval(coeffs[::-1], xv)
        print(f"  x={xv:3.1f}, k_Y(x)={yv: .6g}, poly={f_val: .6g}, error={abs(f_val-yv): .2g}")

    plt.show()

if __name__ == "__main__":
    main()
