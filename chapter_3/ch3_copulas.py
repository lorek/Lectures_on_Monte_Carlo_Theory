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
## File: ch3_copulas.py                                             ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##

import matplotlib.pyplot as plt
import argparse
import os
from scipy.stats import norm
from numpy.random import default_rng, PCG64



def ParseArguments():
    """
    Parse command-line arguments.
    --n: number of replications.
    --results_path: directory for saving plots.
    """
    parser = argparse.ArgumentParser(description="Normal Copula Simulation")
    parser.add_argument('--n_samples', default="200", help='Number of replications (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Directory for saving plots (default: %(default)s)')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()
    return args


def normal_copula(n, rho, rng):
    """
    Generate n samples from a bivariate normal copula with correlation rho.
    This is done by:
      1. Sampling (X, Y) ~ N(0, Σ) with Σ = [[1, rho], [rho, 1]].
      2. Transforming X and Y using the standard normal CDF, so that U1 = Φ(X) and U2 = Φ(Y).
    """

    cov = np.array([[1, rho], [rho, 1]])
    mean = np.array([0, 0])
    samples = rng.multivariate_normal(mean, cov, size=n)
    U1 = norm.cdf(samples[:, 0])
    U2 = norm.cdf(samples[:, 1])
    return U1, U2

def clayton_copula(n, theta,rng):
    """
    Generate n samples from the Clayton copula with parameter theta > 0.
    Uses the algorithm:
      - For each replication, sample W ~ Gamma(1/theta, 1)
      - Sample independent E1, E2 ~ Exp(1)
      - Set U_i = (1 + E_i/W)^(-1/theta), i=1,2.
    """

    W = rng.gamma(shape=1 / theta, scale=1, size=n)
    E1 = rng.exponential(scale=1, size=n)
    E2 = rng.exponential(scale=1, size=n)
    U1 = (1 + E1 / W) ** (-1 / theta)
    U2 = (1 + E2 / W) ** (-1 / theta)
    return U1, U2


import sys
import numpy as np


def frank_copula(n, theta,rng):
    """
    Generate n samples from the Frank copula using the following procedure:

      1. Let U1 and W be independent samples from U(0,1).
      2. If theta == 0, raise an error.
      3. If |theta| > log(max_float), then set:
             U2 = (U1 < 0) + sign(theta) * U1.
      4. Else if |theta| > sqrt(epsilon), compute:
             U2 = -1/theta * log( (exp(-theta * U1) * (1 - W) / W + exp(-theta)) /
                                 (1 + exp(-theta * U1) * (1 - W) / W) ).
      5. Otherwise, set U2 = W.

    Parameters
    ----------
    n : int
        Number of samples.
    theta : float
        Frank copula parameter (must be nonzero).

    Returns
    -------
    U1, U2 : ndarray
        Two arrays of length n, where (U1, U2) are samples from the Frank copula.
    """
    U1 = rng.uniform(size=n)
    W = rng.uniform(size=n)

    if theta == 0:
        raise ValueError("The parameter for Frank copula should not be 0.")

    if abs(theta) > np.log(sys.float_info.max):
        U2 = (U1 < 0) + np.sign(theta) * U1
    elif abs(theta) > np.sqrt(sys.float_info.epsilon):
        numerator = np.exp(-theta * U1) * (1 - W) / W + np.exp(-theta)
        denominator = 1 + np.exp(-theta * U1) * (1 - W) / W
        U2 = -1 / theta * np.log(numerator / denominator)
    else:
        U2 = W

    return U1, U2


def plot_copula(U1, U2, title, filename, results_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-0.005,  1.05)
    ax.set_ylim(-0.005, 1.05)

    ax.scatter(U1, U2, color='blue', alpha=0.75, s=40)


    # Remove top and right spines to show only the x- and y-axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reposition spines so that the axes cross at zero.
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))


    out_path = os.path.join(results_path, filename)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    ax.set_title(title)




######################################
##            Main Code             ##
######################################

def main():
    args = ParseArguments()
    n_samples = int(args.n_samples)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    # Normal copula with rho = 0.9
    U1_normal_09, U2_normal_09 = normal_copula(n=n_samples, rho=0.9,rng=rng)
    plot_copula(U1_normal_09, U2_normal_09,
                "Normal Copula (ρ = 0.9)", "ch3_copulas_normal_rho0.9.pdf", results_path)

    # Normal copula with rho = -0.8
    U1_normal_neg08, U2_normal_neg08 = normal_copula(n=n_samples, rho=-0.8,rng=rng)
    plot_copula(U1_normal_neg08, U2_normal_neg08,
                "Normal Copula (ρ = -0.8)", "ch3_copulas_normal_rho-0.8.pdf", results_path)



    # Clayton Copula: theta = 10 and theta = 2
    U1_clayton_10, U2_clayton_10 = clayton_copula(n=n_samples, theta=10,rng=rng)
    U1_clayton_2, U2_clayton_2 = clayton_copula(n=n_samples, theta=2,rng=rng)

    plot_copula(U1_clayton_10, U2_clayton_10,
                "Clayton Copula (θ = 10)", "ch3_copulas_clayton_theta10.pdf", results_path)
    plot_copula(U1_clayton_2, U2_clayton_2,
                "Clayton Copula (θ = 2)", "ch3_copulas_clayton_theta2.pdf", results_path)

    # Frank Copula: theta = 10 and theta = -20
    U1_frank_10, U2_frank_10 = frank_copula(n=n_samples, theta=10, rng=rng)
    U1_frank_neg20, U2_frank_neg20 = frank_copula(n=n_samples, theta=-20, rng=rng)


    plot_copula(U1_frank_10, U2_frank_10,
                "Frank Copula (θ = 10)", "ch3_copulas_frank_theta10.pdf", results_path)
    plot_copula(U1_frank_neg20, U2_frank_neg20,
                "Frank Copula (θ = -20)", "ch3_copulas_frank_theta-20.pdf", results_path)

    # Normal copula with rho = 0.9
    U1_normal_neg05, U2_normal_neg05 = normal_copula(n=n_samples, rho=-0.5, rng=rng)
    plot_copula(U1_normal_neg05, U2_normal_neg05,
                "Normal Copula (ρ = -0.5)", "ch3_copulas_normal_rho-0.5.pdf", results_path)

    # Normal copula with rho = 0.8
    U1_normal_neg08, U2_normal_neg08 = normal_copula(n=n_samples, rho=0.8, rng=rng)
    plot_copula(U1_normal_neg08, U2_normal_neg08,
                "Normal Copula (ρ = 0.8)", "ch3_copulas_normal_rho0.8.pdf", results_path)


    Exp1_normal_neg05 = -np.log(U1_normal_neg05)
    Par2_normal_neg05 = np.sqrt(U2_normal_neg05 )-1

    #plot_copula(Exp1_normal_neg05,  Par2_normal_neg05,
    #            "ddddd", "ch3_copulas_Exp1_Par2_normal-0.5.pdf", results_path)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(Exp1_normal_neg05, Par2_normal_neg05, color='blue', alpha=0.75, s=40)

    # Remove top and right spines to show only the x- and y-axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Reposition spines so that the axes cross at zero.
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))

    plt.show()

if __name__ == "__main__":
    main()
