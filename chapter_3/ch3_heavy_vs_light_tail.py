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
## File: ch3_heavy_vs_light_tail.py                                 ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from numpy.random import default_rng, PCG64

def ParseArguments():
    parser = argparse.ArgumentParser(description="Compare distributions via random walk and time series plots.")
    parser.add_argument('--results_path', default="results", help="Output folder")
    parser.add_argument('--N', default="4500", help="Random seed")
    parser.add_argument('--seed', default="31415", help="Random seed")
    return parser.parse_args()


def draw_samples(rng, dist_type, n):
    if dist_type == 'exp':
        return rng.exponential(scale=1.0, size=n)
    elif dist_type == 'normal':
        return rng.normal(loc=1.0, scale=1.0, size=n)
    elif dist_type == 'pareto':
        return 0.2 * rng.pareto(a=1.2, size=n)
    elif dist_type == 'lognormal':
        return np.exp(rng.normal(loc=-0.5, scale=1.0, size=n))
    elif dist_type == 'weibull':
        U = rng.uniform(0, 1, size=n)
        return (-np.log(U) / np.sqrt(2)) ** 2
    else:
        raise ValueError("Unknown distribution: " + dist_type)


def sample_random_walk(X):
    S = np.cumsum(X)
    return S / np.arange(1, len(X) + 1)


def plot_walks(X1, X2, X3, labels, filename, results_path):
    plt.figure(figsize=(5.6, 4))
    for X, label, color in zip([X1, X2, X3], labels, ['green', 'blue', 'red']):
        RW = sample_random_walk(X)
        plt.plot(np.arange(1, len(X) + 1), RW, label=label, color=color)
    plt.legend()
    plt.savefig(os.path.join(results_path, filename), bbox_inches='tight', pad_inches=0)
    print("Saved figure: ", os.path.join(results_path, filename))
    plt.xlabel("n")
    plt.ylabel("S_n / n")
    plt.title("Random Walk Averages: S_n / n")


def plot_time_series(X1, X2, X3, labels, filename, results_path):
    X_ts = np.concatenate([X1, X2, X3])
    n_block = len(X1)
    x1 = np.arange(1, n_block + 1)
    x2 = np.arange(n_block + 1, 2 * n_block + 1)
    x3 = np.arange(2 * n_block + 1, 3 * n_block + 1)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_ylim(1.1 * np.min(X_ts), 25)
    plt.plot(x1, X1, lw=1.0, color='green', label=labels[0])
    plt.plot(x2, X2, lw=1.0, color='blue', label=labels[1])
    plt.plot(x3, X3, lw=1.0, color='red', label=labels[2])
    plt.legend()
    plt.savefig(os.path.join(results_path, filename), bbox_inches='tight', pad_inches=0)
    print("Saved figure: ",os.path.join(results_path, filename))
    plt.xlabel("Index")
    plt.ylabel("X_i")
    plt.title("Piecewise Time Series")
    plt.legend()


def main():
    args = ParseArguments()
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = default_rng(PCG64(seed))

    N = int(args.N)

    # --- Old distributions ---
    X_exp = draw_samples(rng, 'exp', N)
    X_par = draw_samples(rng, 'pareto', N)
    X_norm = draw_samples(rng, 'normal', N)


    plot_walks(X_norm, X_exp, X_par,
               labels=["N(1,1)", "Exp(1)", "0.2*Pareto(1.2)" ],
               filename="ch3_heavy_vs_light_tail_walk_exp_par_norm.pdf",
               results_path=results_path)

    plot_time_series(X_norm, X_exp, X_par,
                     labels=["N(1,1)", "Exp(1)", "0.2*Pareto(1.2)", ],
                     filename="ch3_heavy_vs_light_tail_ts_exp_par_norm.pdf",
                     results_path=results_path)

    # --- New distributions ---
    X_log = draw_samples(rng, 'lognormal', N)
    X_par2 = draw_samples(rng, 'pareto', N)
    X_wei = draw_samples(rng, 'weibull', N)



    plot_walks(X_log,  X_wei, X_par2,
               labels=["exp(N(-0.5,1))", "(-log(U)/sqrt(2))^2","0.2*Pareto(1.2)"],
               filename="ch3_heavy_vs_light_tail_walk_log_par_weib.pdf",
               results_path=results_path)

    plot_time_series(X_log, X_wei, X_par2,
                     labels=["exp(N(-0.5,1))", "(-log(U)/sqrt(2))^2", "0.2*Pareto(1.2)"],
                     filename="ch3_heavy_vs_light_tail_ts_log_par_weib.pdf",
                     results_path=results_path)

    #print(f"All plots saved in folder: {results_path}")

    print("\nMaximal values: ")
    print("max(X_norm) = ", np.round(np.max(X_norm),2), ",\t max(X_exp) = ", np.round(np.max(X_exp),2), ", \t max(X_par) = ", np.round(np.max(X_par),2))
    print("max(X_log) = ", np.round(np.max(X_log),2), ", \t max(X_wei) = ", np.round(np.max(X_wei),2), ",\t max(X_par2 ) = ", np.round(np.max(X_par2),2))

    plt.show()


if __name__ == "__main__":
    main()
