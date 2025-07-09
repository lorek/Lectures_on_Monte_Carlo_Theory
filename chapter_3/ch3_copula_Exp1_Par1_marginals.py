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
## File: ch3_copula_Exp1_Par1_marginals.py                          ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.stats import norm, pareto, gamma


def ParseArguments():
    parser = argparse.ArgumentParser(
        description="Sampling from a joint density f(x,y) = x exp(-x(y+1)) and copula transformations")
    parser.add_argument('--n', default="100", help="Number of samples (default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Directory for saving plots (default: %(default)s)")
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None' for no fixed seed. (default: %(default)s)")
    # New argument: rho values as comma-separated list (default: "0.5,0.8")
    parser.add_argument('--rho', default="-0.5,0.8",
                        help="Comma-separated list of correlation values for the normal copula (default: %(default)s)")
    args = parser.parse_args()
    return args


def sample_joint(n, rng):
    """
    Sample n points from the joint density:
      f(x,y) = x exp(-x(y+1)) for x, y >= 0.
    Marginal for X is Exp(1): X = -log(U1)
    Conditional for Y given X=x is Exp(x): Y = -log(U2)/x.
    """
    U1 = rng.uniform(0, 1, n)
    X = -np.log(U1)  # X ~ Exp(1)
    U2 = rng.uniform(0, 1, n)
    Y = -np.log(U2) / X  # Y|X=x ~ Exp(x)
    return X, Y


def normal_copula(n, rho, rng):
    """
    Generate n samples from a bivariate normal copula with correlation rho.
    This is done by:
      1. Sampling (X, Y) ~ N(0, Σ) with Σ = [[1, rho], [rho, 1]].
      2. Transforming X and Y using the standard normal CDF, so that
         V1 = Φ(X) and V2 = Φ(Y).
    """
    cov = np.array([[1, rho], [rho, 1]])
    mean = np.array([0, 0])
    samples = rng.multivariate_normal(mean, cov, size=n)
    # Rename the transformed variables to V1 and V2
    V1 = norm.cdf(samples[:, 0])
    V2 = norm.cdf(samples[:, 1])
    return V1, V2


def transform_copula_to_marginals(V1, V2):
    """
    Transform copula samples (V1, V2) to (X, Y) with:
      X ~ Exp(1) -> X = -log(1-V1)
      Y ~ Par(1) -> Y = 1/(1-V2) - 1
    """
    X = -np.log(1 - V1)  # X ~ Exp(1)
    Y = 1 / (1 - V2) - 1  # Y ~ Par(1)
    return X, Y


def plot_joint(X, Y, results_path, filename, title, color="blue"):
    """
    Scatter plot of (X, Y) with appropriate formatting.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.05, np.max(X) * 1.05)
    ax.set_ylim(-0.1, 10.15)
    ax.scatter(X, Y, color=color, alpha=0.75, s=40)

    # Reposition spines so that the axes cross at zero.
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    plt.savefig(os.path.join(results_path, filename), bbox_inches='tight', pad_inches=0)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (Exp(1) marginal)")
    ax.set_ylabel("Y (Pareto(2) marginal)")
    print("Saved ",os.path.join(results_path, filename))


def main():
    args = ParseArguments()
    n = int(args.n)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Set seed: if user provided "none" then no fixed seed.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    # (1) Generate samples from the joint density f(x,y)
    X, Y = sample_joint(n, rng)
    plot_joint(X, Y, results_path, "ch3_copula_Exp1_Par1_joint_scatter.pdf",
               "Scatter plot of (X,Y) from joint density")

    # (2) Process each value of rho provided on the command line.
    # rho argument is a comma-separated list (default "0.5,0.8")
    # Process each value of rho provided on the command line.
    rho_str_list = [s.strip() for s in args.rho.split(',') if s.strip()]
    rho_list = [float(s) for s in rho_str_list]

    # For each rho, choose a color based on its value.
    for rho in rho_list:
        # Choose color: if rho is negative, use green; if nonnegative, use blue.
        color = "green" if rho < 0 else "blue"

        # Generate samples using normal copula with current rho.
        V1, V2 = normal_copula(n, rho, rng)

        # Create a string representation for rho with decimal replaced by an underscore.
        rho_str = f"{rho:.2f}".replace('.', '_')

        # Save copula scatter plot (of (V1, V2)) using the chosen color.
        copula_filename = f"ch3_copula_Exp1_Par1_normal_rho{rho_str}_V1V2.pdf"
        plt.figure(figsize=(6, 6))
        plt.scatter(V1, V2, color=color, alpha=0.75, s=40)
        plt.grid(True)
        plt.savefig(os.path.join(results_path, copula_filename), bbox_inches='tight', pad_inches=0)
        plt.title(f"Normal copula with ρ = {rho}", fontsize=12)
        plt.xlabel("V1 (Φ(X))", fontsize=10)
        plt.ylabel("V2 (Φ(Y))", fontsize=10)
        print("Saved ", os.path.join(results_path, copula_filename))

        # Transform copula samples to marginals (X, Y).
        X_trans, Y_trans = transform_copula_to_marginals(V1, V2)

        # Save scatter plot of (X, Y) after transformation.
        marginals_filename = f"ch3_copula_Exp1_Par1_normal_rho{rho_str}.pdf"
        # Pass the chosen color to plot_joint.
        plot_joint(X_trans, Y_trans, results_path, marginals_filename,
                   f"Scatter plot of (X,Y) using normal copula (ρ = {rho})", color=color)

    plt.show()


if __name__ == "__main__":
    main()
