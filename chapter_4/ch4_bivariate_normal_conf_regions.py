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
## File: ch4_bivariate_normal_conf_regions.py                       ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




"""
Confidence Regions for a Bivariate Normal Distribution

This script simulates 5000 samples from a bivariate normal distribution
with mean zero and Sigmaariance matrix Σ, and computes three types of confidence regions at level 1−α:
  - An ellipsoidal region: { x : xᵀ Σ⁻¹ x ≤ q },
  - A parallelogram obtained by mapping the hypercube [-z, z]² via the square root of Σ,
  - A rectangular region based on the marginal standard deviations.
Parameters (α and Σ) are provided via command-line arguments.
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
import matplotlib.transforms as transforms
import argparse
import os
from tabulate import tabulate
from scipy.stats import chi2
from scipy.linalg import sqrtm
from scipy.stats import norm
from numpy.random import default_rng, PCG64

def ParseArguments():
    parser = argparse.ArgumentParser(description="Confidence regions for bivariate normal data.")
    parser.add_argument('--alpha', default="0.05", help="Significance level alpha (default: %(default)s)")
    # Sigma provided as comma-separated values: a,b,c,d to form 2x2 matrix [[a,b],[c,d]]
    parser.add_argument('--Sigma', default="1,2,2,6", help="Covariance matrix (a,b,b,d)=[[a,b],[b,d]] entries (default: %(default)s)")
    parser.add_argument('--n_samples', default="2000", help="Number of samples (default: %(default)s)")
    parser.add_argument('--results_path', default="results", help="Folder for saving outputs (default: %(default)s)")
    parser.add_argument('--seed', default="31415", help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")
    args = parser.parse_args()

    return args


def main():
    # Parse input parameters
    args = ParseArguments()

    alpha = float(args.alpha)
    # Parse Sigma string into 2x2 matrix
    sigma_vals = [float(v) for v in args.Sigma.split(",")]
    Sigma = np.array([[sigma_vals[0], sigma_vals[1]],
                      [sigma_vals[2], sigma_vals[3]]])

    n_samples = int(args.n_samples)
    results_path = args.results_path
    os.makedirs(results_path, exist_ok=True)

    # Check symmetry
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Sigma is not symmetric. Please provide a valid covariance matrix.")

    # Check positive definiteness (all eigenvalues > 0)
    if np.any(np.linalg.eigvals(Sigma) <= 0):
        raise ValueError("Sigma is not positive definite. Please provide a valid covariance matrix.")

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    # Set mean to zero.
    mu = np.array([0.0, 0.0])
    # Simulate n_samples from N(0, Sigma)
    X = rng.multivariate_normal(mu, Sigma, size=n_samples)



    # Separate components: let X[:,0] be weight (denoted as hat{Y}^1_R) and X[:,1] be height (hat{Y}^2_R)
    Y1 = X[:, 0]
    Y2 = X[:, 1]

    # Compute sample means and standard deviations
    Y1_mean = np.mean(Y1)
    Y1_std = np.std(Y1, ddof=1)
    Y2_mean = np.mean(Y2)
    Y2_std = np.std(Y2, ddof=1)

    CI_Y1 = (Y1_mean - 1.96 * Y1_std / np.sqrt(n_samples), Y1_mean + 1.96 * Y1_std / np.sqrt(n_samples))
    CI_Y2 = (Y2_mean - 1.96 * Y2_std / np.sqrt(n_samples), Y2_mean + 1.96 * Y2_std / np.sqrt(n_samples))

    table_stats = [
        ["", "Mean", "Std Dev", "95% CI"],
        [r"hat{Y}^1_R", f"{Y1_mean:.4f}", f"{Y1_std:.4f}", f"({CI_Y1[0]:.4f}, {CI_Y1[1]:.4f})"],
        [r"\hat{Y}^2_R", f"{Y2_mean:.4f}", f"{Y2_std:.4f}", f"({CI_Y2[0]:.4f}, {CI_Y2[1]:.4f})"]
    ]
    print("\n--- Summary Statistics ---")
    print(tabulate(table_stats, headers="firstrow", tablefmt="grid"))


    print("\nCovariance matrix (Sigma):")
    print(Sigma)

    # Compute eigen-decomposition of Sigma (or Sigma)
    eig_vals, eig_vecs = np.linalg.eig(Sigma)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    print("\nEigenvalues:")
    print(eig_vals)
    print("\nEigenvectors (columns correspond to eigenvalues):")
    print(eig_vecs)

    # Compute chi-square quantile for 2 degrees of freedom at 1-alpha.
    q_val = chi2.ppf(1 - alpha, df=2)  # e.g. for alpha=0.05, q_val ~ 5.991
    print("alpha = ",alpha)
    print(f"\nChi-square quantile (df=2, 1-alpha): {q_val:.4f}")


    # Confidence Ellipse: points satisfying x^T Sigma^{-1} x <= q_val.
    # The ellipse has principal axes lengths: 2*sqrt(q_val * lambda_i) (i=1,2)
    major_len = 2.0 * np.sqrt(eig_vals[0] * q_val)
    minor_len = 2.0 * np.sqrt(eig_vals[1] * q_val)
    angle_deg = np.degrees(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))
    ellipse_patch = Ellipse(xy=(Y1_mean, Y2_mean),
                            width=major_len, height=minor_len,
                            angle=angle_deg,
                            edgecolor='black', facecolor='gray', alpha=0.4, lw=2, label="Confidence Ellipse")


    print("\n**** ELLIPSOID  confidence region  ****")
    print("\n")
    g1 = -(major_len / 2) * eig_vecs[:, 0]
    g2 = (minor_len / 2) * eig_vecs[:, 1]
    print("Principal axes: ")
    print("g1 = ", g1, ", of length = ", major_len / 2 )
    print("g2 = ", g2, ", of length = ", minor_len / 2 )


    print("\n**** Parallelogram  confidence region  ****")
    print("\n")
    # Confidence Parallelogram: Compute square root S of Sigma.
    S = sqrtm(Sigma)  # symmetric square root
    # Compute hypercube vertices in 2D: using z_val from the alternative formula
    z_val = norm.ppf((np.sqrt(1 - alpha) + 1) / 2)
    # Compute the alternative z value: z = Phi^{-1}(((sqrt(1-alpha)+1)/2))
    print("z_val  = Phi^{-1}(((sqrt(1-alpha)+1)/2) =  ",z_val)
    v_corners = np.array([
        [-z_val, -z_val],
        [-z_val, z_val],
        [z_val, z_val],
        [z_val, -z_val]
    ])

    # Map vertices via S and shift by mean:
    parallelogram_corners = np.array([(S @ v) + np.array([Y1_mean, Y2_mean]) for v in v_corners])
    parallelogram_patch = Polygon(parallelogram_corners, closed=True, edgecolor='brown', facecolor='none', lw=2,
                                  label="Parallelogram")


    print("parallelogram corners:")
    print(parallelogram_corners)
    # Rectangular Confidence Region: using marginal stds.
    rect_width = 2 * Y1_std * z_val
    rect_height = 2 * Y2_std * z_val



    print("\n**** Rectangular confidence region  ****")
    print("\n")
    c1 = np.round(np.sqrt(Sigma[0, 0]) * z_val, 4)
    c2 = np.round(np.sqrt(Sigma[1, 1]) * z_val, 4)

    print("c1 = sigma1 * ",np.round(z_val,4), " = sqrt(",Sigma[0,0],") *",np.round(z_val,4), "=", c1)
    print("c2 = sigma2 * ",np.round(z_val,4), " = sqrt(",Sigma[1,1],") *",np.round(z_val,4), "=", c2)
    print("C_rec = [",-c1, ",", c1,"]x[",-c2,",",c2,"]")


    # Lower left corner is (Y1_mean - rect_width/2, Y2_mean - rect_height/2)
    rectangle_patch = plt.Rectangle((Y1_mean - rect_width / 2, Y2_mean - rect_height / 2),
                                    rect_width, rect_height,
                                    edgecolor='blue', facecolor='none', linestyle='-', lw=1, alpha=0.6, label="Rectangle")


    # Plot figure: Scatter 5000 points and overlay regions.
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(np.min(Y1)*1.2, np.max(Y1)*1.2)
    ax.set_ylim(np.min(Y2)*0.95, np.max(Y2)*0.95)

    ax.scatter(Y1, Y2, s=20, color='black', alpha=0.2, label="Samples")
    ax.set_aspect('equal')
    ax.add_patch(ellipse_patch)
    ax.add_patch(parallelogram_patch)
    ax.add_patch(rectangle_patch)


    # 2. Place the spines at zero
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # 3. Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 4. Adjust ticks so that they appear only at the bottom and left
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Draw principal axes as arrows from the center.
    origin = np.array([Y1_mean, Y2_mean])
    # Principal axes are half the ellipse's axis lengths.

    ax.arrow(origin[0], origin[1], g1[0], g1[1],
             head_width=0.2, head_length=0.4, alpha=0.8, fc='black', ec='black',
             lw=1, length_includes_head=True, label="Principal Axis 1")
    ax.arrow(origin[0], origin[1], g2[0], g2[1],
             head_width=0.2, head_length=0.4, alpha=0.8, fc='black', ec='black',
             lw=1, length_includes_head=True, label="Principal Axis 2")


    plt.tight_layout()
    fig_path = os.path.join(results_path, "ch4_bivariate_normal_conf_regions.pdf")
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    print(f"\nPlot saved to: {fig_path}")

    ax.set_title("Confidence Regions for Bivariate Normal Data")

    plt.show()


if __name__ == "__main__":
    main()
