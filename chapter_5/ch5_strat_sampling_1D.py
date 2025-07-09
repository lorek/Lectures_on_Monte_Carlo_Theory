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
## File: ch5_strat_sampling_1D.py                                   ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Stratified and Traditional Sampling from N(0,1)

This script produces two types of plots:
1. A CDF plot showing the standard normal c.d.f. with sample quantiles from a stratified sample.
2. A comparison of histograms for N(0,1) drawn using stratified sampling and traditional sampling.
The figures are saved as PDFs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
import os


def ParseArguments():
    """
    Parse command-line arguments.
    --n: Total number of samples.
    --m_cdf: Number of strata for computing the sample quantiles (CDF plot).
    --m_hist: Number of strata for stratified histogram.
    --bins: Number of bins for histogram.
    --results_path: Folder to save the resulting figures.
    """
    parser = argparse.ArgumentParser(description="Sampling from N(0,1) with stratification")
    parser.add_argument('--n', default="500", help='Total number of samples (default: %(default)s)')
    parser.add_argument('--m_cdf', default="10", help='Number of strata for CDF (default: %(default)s)')
    parser.add_argument('--m_hist', default="100", help='Number of strata for histogram (default: %(default)s)')
    parser.add_argument('--bins', default="50", help='Number of histogram bins (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Folder for saving plots (default: %(default)s)')
    parser.add_argument('--seed', default="31415", help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()
    return args


def main():

    args = ParseArguments()
    n = int(args.n)
    m_cdf = int(args.m_cdf)
    m_hist = int(args.m_hist)
    nr_bins = int(args.bins)

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)



    ###########################################
    # 1) CDF Plot with Stratified Quantiles   #
    ###########################################

    # Compute stratification levels for CDF plot: divide (0,1) into m_cdf equal strata.
    p_levels = np.arange(1, m_cdf) / m_cdf
    # Obtain quantiles of the standard normal distribution.
    quantiles = norm.ppf(p_levels)

    # Create a figure for the CDF.
    fig_cdf, ax_cdf = plt.subplots(figsize=(10, 4))
    x_vals = np.arange(-3, 3.4, 0.1)

    # Remove the default spines
    ax_cdf.spines['top'].set_visible(False)
    ax_cdf.spines['right'].set_visible(False)
    ax_cdf.spines['bottom'].set_visible(False)
    ax_cdf.spines['left'].set_visible(False)

    # Turn off the default ticks
    ax_cdf.tick_params(bottom=False, left=False, labelbottom=True, labelleft=True)

    # Draw custom arrowed x-axis and y-axis
    xmin, xmax = x_vals.min(),  x_vals.max()
    print(" xmax = ", xmax, " ty = ", type(xmax))
    xmax=3.2
    print(" xmax = ", xmax, " ty = ", type(xmax))

    ax_cdf.annotate("", xy=(xmax, 0), xytext=(xmin, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1,
                                     shrinkA=0, shrinkB=0, mutation_scale=20))
    ax_cdf.annotate("", xy=(0, 1), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1,
                                     shrinkA=0, shrinkB=0, mutation_scale=20))


    ax_cdf.plot(x_vals, norm.cdf(x_vals), label="Standard Normal CDF", color='black')
    # Mark the stratification boundaries on the CDF
    ax_cdf.plot(quantiles, np.zeros_like(quantiles), 'r+', markersize=8)
    for q in quantiles:
        # Draw a small orange line from the quantile to the c.d.f. value at that point.
        ax_cdf.plot([q, q, 0], [0, norm.cdf(q), norm.cdf(q)], color='tab:gray', linestyle="--", lw=1)


    # ax_cdf.axhline(y=0, color='gray', lw=1)
    # ax_cdf.axvline(x=0, color='gray', lw=1)
    ax_cdf.set_ylim(0, 1)

    cdf_fig_path = os.path.join(results_path, "ch5_strat_sampling_1D_cdf.pdf")
    plt.savefig(cdf_fig_path, bbox_inches='tight', pad_inches=0)
    ax_cdf.set_title("Standard Normal CDF with Stratified (m=" + str(m_cdf)+" strata) Quantiles")
    ax_cdf.set_xlabel("x")
    ax_cdf.set_ylabel("F(x)")
    print(f"CDF figure saved to: {cdf_fig_path}")

    ###########################################
    # 2) Histogram Comparison: Stratified vs Traditional
    ###########################################

    # 2a) Stratified Sampling:
    # Divide [0,1) into m_hist strata; in each stratum, sample ni = n/m_hist points.
    U_strat = np.array([])
    X_strat = np.array([])

    for i in range(m_hist):
        ni = int(n / m_hist)
        U_i = rng.uniform(0, 1, ni)
        # Map U_i to the stratum: V = i/m_hist + U_i/m_hist
        V_i = i / m_hist + U_i / m_hist
        # Compute X from the standard normal using the inverse CDF on V
        X_i = norm.ppf(V_i)
        U_strat = np.concatenate((U_strat, U_i))
        X_strat = np.concatenate((X_strat, X_i))

    # 2b) Traditional Sampling:
    U_trad = rng.uniform(0, 1, n)
    X_trad = norm.ppf(U_trad)

    # Compute summary statistics
    mean_strat = np.round(np.mean(X_strat), 4)
    var_strat = np.round(np.var(X_strat, ddof=1), 4)
    mean_trad = np.round(np.mean(X_trad), 4)
    var_trad = np.round(np.var(X_trad, ddof=1), 4)

    # Create a figure with two subplots for histograms.
    fig_hist, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

    ax_left.hist(X_strat, bins=nr_bins, density=True, edgecolor='black', facecolor='blue', alpha=0.75)
    ax_left.plot(x_vals, norm.pdf(x_vals), color='red')
    ax_left.set_ylim(0, 0.5)

    ax_right.hist(X_trad, bins=nr_bins, density=True, edgecolor='black', facecolor='blue', alpha=0.6)

    ax_right.plot(x_vals, norm.pdf(x_vals), color='red')
    ax_right.set_ylim(0, 0.5)

    hist_fig_path = os.path.join(results_path, "ch5_strat_sampling_1D_hist.pdf")
    plt.savefig(hist_fig_path, bbox_inches='tight', pad_inches=0)
    print(f"Histogram figure saved to: {hist_fig_path}")

    fig_hist.suptitle(f"Histogram of {n} Samples from N(0,1) with {nr_bins} bins")
    ax_left.set_title(f"Stratified Sampling\nMean = {mean_strat}, Var = {var_strat}")
    ax_right.set_title(f"Traditional Sampling\nMean = {mean_trad}, Var = {var_trad}")


    plt.show()

if __name__ == '__main__':
    main()
