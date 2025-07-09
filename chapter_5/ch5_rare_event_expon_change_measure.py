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
## File: ch5_rare_event_expon_change_measure.py                     ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Plotting and Simulation for Exponential Change of Measure and Rare Event Estimation

This script performs simulations for estimating I(n) = P(Sn > n(μ+ε)) for a coin-flipping model.
For various choices of the importance sampling parameter (θ) – including the optimal value θ₀ and
the CMC case (represented by θ = -1) – it computes the estimator and its relative variance (Var(Y)/I²),
and plots:
  1. Ratio Var(Y^θ(n))/(I_θ(n))² versus n.
  2. Log of the above ratio.
  3. The estimator Ŷ_R^θ (or its log, if requested).
  4. Confidence intervals for the relative error and the absolute relative error.

Each figure is saved as a PDF in the specified results folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
import argparse
import os
import math
from numpy.random import default_rng, PCG64

# For nicer plots, you can enable LaTeX text rendering if desired.
# plt.rcParams['text.usetex'] = True

def ParseArguments():
    parser = argparse.ArgumentParser(description="Rare Event Simulation and Plotting for Exponential Change of Measure")
    parser.add_argument('--n_min', default="2", help="Minimum n (number of coin flips), default: %(default)s")
    parser.add_argument('--n_max', default="50", help="Maximum n (number of coin flips), default: %(default)s")
    parser.add_argument('--n_step', default="1", help="Step size for n, default: %(default)s")
    parser.add_argument('--R', default="5000", help="Number of replications, default: %(default)s")
    parser.add_argument('--eps', default="0.1", help="Epsilon (shift parameter), default: %(default)s")
    parser.add_argument('--seed', default="31415", help="Seed for the PRNG (default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Folder for saving output figures (default: %(default)s)")
    parser.add_argument('--plot_log', action="store_true", help="Plot the estimator on a logarithmic scale")
    return parser.parse_args()

def compute_kappa(theta):
    return np.log((1 + np.exp(theta)) / 2.0)

def main():
    args = ParseArguments()
    n_min = int(args.n_min)
    n_max = int(args.n_max)
    n_step = int(args.n_step)
    R = int(args.R)
    eps = float(args.eps)
    seed = None if args.seed.lower() == "none" else int(args.seed)
    results_path = args.results_path
    plot_log = args.plot_log

    os.makedirs(results_path, exist_ok=True)
    rng = default_rng(PCG64(seed=seed))

    mu = 0.5

    # Compute the optimal theta (θ₀) from theory.
    theta0 = np.log(1 + 2 * eps) - np.log(1 - 2 * eps)
    print("Optimal theta (theta0) computed as:", theta0)

    # Define the list of theta values.
    # We'll use theta0, one alternative value (1.4) and the CMC case (theta = -1)
    thetas = [theta0, 1.4, -1]

    # Build labels for the legend.
    thetas_text = []
    for idx, th in enumerate(thetas):
        if idx == 0:
            thetas_text.append("θ₀=" + str(np.round(th, 3)))
        elif th == -1:
            thetas_text.append("CMC")
        else:
            thetas_text.append(str(np.round(th, 3)))

    # Precompute the true probability I_true(n) for each n using the binomial distribution.
    print("Now computing true probabilities I(n) for n in range [{}, {})...".format(n_min, n_max))
    n_values = np.arange(n_min, n_max, n_step)
    I_true = []
    for n in n_values:
        # Compute threshold: if n*(mu+eps) is integer, we need to take the next integer.
        a = n * (mu + eps)
        b = np.ceil(a)
        c = a + 1 if a == b else b
        prob = 0.0
        for k in np.arange(c, n + 1):
            prob += binom.pmf(k, n, 0.5)
        I_true.append(prob)
    I_true = np.array(I_true)

    # Prepare arrays to store simulation results for each theta.
    y_est_all = []  # estimated I(n)
    y_val_all = []  # relative variance: Var(Y)/[I(n)]^2

    # Loop over each theta value.
    for idx, theta in enumerate(thetas):
        print("Now simulating for theta =", theta, "(", thetas_text[idx], ")")
        y_est = []  # estimator for each n
        y_val = []  # relative variance for each n

        for n in n_values:
            # Determine the coin probabilities.
            if theta == -1:
                # CMC case: fair coin
                probs = [0.5, 0.5]
            else:
                # Under exponential change of measure, probabilities become:
                # p(0) = 1/(1+exp(theta)), p(1) = exp(theta)/(1+exp(theta))
                probs = [1 / (1 + np.exp(theta)), np.exp(theta) / (1 + np.exp(theta))]

            # Simulation: generate R samples of n coin flips.
            # Using rng.choice with provided probabilities.
            Sn = np.empty(R, dtype=int)
            for i in range(R):
                sample = rng.choice([0, 1], size=n, p=probs)
                Sn[i] = np.sum(sample)

            # Indicator for the event {Sn > n*(mu+eps)}
            indicator = (Sn > n * (mu + eps)).astype(float)

            if theta != -1:
                kappa = compute_kappa(theta)
                # Compute the likelihood ratio weight: exp(-theta * Sn + n*kappa)
                weights = np.exp(-theta * Sn + n * kappa)
                YY = indicator * weights
            else:
                # For CMC, simply use the indicator.
                YY = indicator

            YR = np.mean(YY)
            VarYR = np.var(YY, ddof=1)
            rel_var = VarYR / (YR**2) if YR != 0 else np.nan

            y_est.append(YR)
            y_val.append(rel_var)
        y_est = np.array(y_est)
        y_val = np.array(y_val)
        print("Completed simulations for", thetas_text[idx])
        y_est_all.append(y_est)
        y_val_all.append(y_val)

    # --- Plotting Section ---
    # Figure 1: Ratio of Variance (Var(Y)/I^2) versus n.
    fig1, ax1 = plt.subplots(figsize=(9, 4.5))

    for idx, y_val in enumerate(y_val_all):
        ax1.plot(n_values, y_val, label=thetas_text[idx])
        ax1.scatter(n_values, y_val, s=10)
    ax1.legend()
    out_file = os.path.join(results_path, "figure1.pdf")
    plt.savefig(out_file, bbox_inches='tight')
    ax1.set_xlabel("n")
    ax1.set_ylabel("Var(Y)/I²")
    ax1.set_title("Ratio of Variance (Var(Y^θ(n))/(I_θ(n))²)")
    print("Saved figure to", out_file)

    # Figure 1b: Log of the ratio.
    fig1b, ax1b = plt.subplots(figsize=(9, 4.5))

    for idx, y_val in enumerate(y_val_all):
        ax1b.plot(n_values, np.log(y_val), label=thetas_text[idx])
        ax1b.scatter(n_values, np.log(y_val), s=10)
    ax1b.legend()
    out_file = os.path.join(results_path, "ch5_rare_event_ratio_log.pdf")
    plt.savefig(out_file, bbox_inches='tight')
    ax1b.set_xlabel("n")
    ax1b.set_ylabel("log(Var(Y)/I²)")
    ax1b.set_title("Logarithm of the Ratio of Variance (Var(Y^θ(n))/(I_θ(n))²)")
    print("Saved figure  to", out_file)

    # Figure 2: Estimation of I (or log(I)) versus n.
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))


    for idx, y_est in enumerate(y_est_all):
        if plot_log:
            ax2.plot(n_values, np.log(y_est), label=thetas_text[idx])
            ax2.scatter(n_values, np.log(y_est), s=10)
        else:
            ax2.plot(n_values, y_est, label=thetas_text[idx])
            ax2.scatter(n_values, y_est, s=10)
    # Plot true I(n) in gray.
    if plot_log:
        ax2.plot(n_values, np.log(I_true), color='gray', linewidth=4.0, alpha=0.5, label="True I")
        ax2.scatter(n_values, np.log(I_true), color='gray', s=12, alpha=0.5)
    else:
        ax2.plot(n_values, I_true, color='gray', linewidth=4.0, alpha=0.5, label="True I")
        ax2.scatter(n_values, I_true, color='gray', s=12, alpha=0.5)
    ax2.legend()
    out_file = os.path.join(results_path, "ch5_rare_event_estim.pdf")
    ax2.set_xlabel("n")
    plt.savefig(out_file, bbox_inches='tight')
    if plot_log:
        ax2.set_ylabel("log(Estimator)")
        ax2.set_title("Log of Estimator Ŷ_R^θ of I")
    else:
        ax2.set_ylabel("Estimator")
        ax2.set_title("Estimator Ŷ_R^θ of I")
    print("Saved figure  to", out_file)

    # Figure 3: Confidence intervals for relative error.
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    # For each theta, we display the confidence interval band (using 1.96/sqrt(R)*sqrt(rel_var))
    for idx, y_val in enumerate(y_val_all):
        ci = 1.96 / np.sqrt(R) * np.sqrt(y_val)
        ax3.fill_between(n_values, -ci, ci, label=thetas_text[idx], alpha=0.2)
    ax3.legend()
    out_file = os.path.join(results_path, "ch5_rare_event_conf_interval.pdf")
    plt.savefig(out_file, bbox_inches='tight')
    ax3.set_xlabel("n")
    ax3.set_ylabel("Relative Error")
    ax3.set_title("Confidence Intervals for Relative Error")
    print("Saved figure   to", out_file)

    # Figure 4: Absolute relative error (|(Estimator - True)/True|) versus n.
    fig4, ax4 = plt.subplots(figsize=(9, 4.5))

    for idx, y_est in enumerate(y_est_all):
        rel_error = np.abs((y_est - I_true) / I_true)
        ax4.plot(n_values, rel_error, label=thetas_text[idx], alpha=0.6)
        ax4.scatter(n_values, rel_error, s=12)
    ax4.legend()
    out_file = os.path.join(results_path, "ch5_rare_event_abs_error.pdf")
    plt.savefig(out_file, bbox_inches='tight')
    ax4.set_xlabel("n")
    ax4.set_ylabel("Absolute Relative Error")
    ax4.set_title("Absolute Relative Error")
    print("Saved figure   to", out_file)

    # Finally, show the plots (optional)
    plt.show()

if __name__ == "__main__":
    main()
