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
## File: ch2_prng_sets_A_B_C.py                                     ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy as cp
from scipy.stats import chisquare, kstest, kstwobign
from scipy.stats import qmc  # For Sobol
from tabulate import tabulate
import argparse
from numpy.random import default_rng, PCG64
import os

# ---------------------------- Helper Functions ---------------------------- #



def ParseArguments():
    parser = argparse.ArgumentParser(description="Goodness-of-fit tests")
    parser.add_argument('--n', default="50", help='Number of points (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Path for saving results (default: %(default)s)')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")
    args = parser.parse_args()
    return args

def ecdf(data):
    """Compute the empirical CDF."""
    n = len(data)
    sorted_data = np.sort(data)
    cdf_vals = np.arange(1, n+1) / n
    return sorted_data, cdf_vals

def compute_counts(partition, data):
    """
    Compute counts of data in each bin defined by partition.
    The first bin is (0, partition[1]) and subsequent bins are [partition[i], partition[i+1]).
    """
    counts = np.zeros(len(partition)-1, dtype=int)
    for i in range(len(counts)):
        lower = partition[i]
        upper = partition[i+1]
        if i == 0:
            counts[i] = np.sum((data > lower) & (data <= upper))
        else:
            counts[i] = np.sum((data >= lower) & (data < upper))
    return counts

def compute_KS_statistic(data, n):
    """Compute the KS statistic D_n for data against Uniform(0,1)."""
    sorted_data = np.sort(data)
    diff_plus  = np.arange(1, n+1)/n - sorted_data
    diff_minus = sorted_data - np.arange(0, n)/n
    return np.maximum(np.max(diff_plus), np.max(diff_minus))

def ks_pvalue(D, n):
    """Compute the KS p-value using the correction:
       p = 1 - K( sqrt(n)*D + 1/(6sqrt(n)) + (sqrt(n)*D - 1)/(4n) )
    """
    arg = np.sqrt(n)*D + 1/(6*np.sqrt(n)) + (np.sqrt(n)*D - 1)/(4*n)
    return 1 - kstwobign.cdf(arg)

def format_partition(partition):
    """Return a list of strings describing each partition interval."""
    intervals = []
    intervals.append(f"P1 = (0, {partition[1]:.2f})")
    for i in range(1, len(partition)-1):
        intervals.append(f"P{i+1} = [{partition[i]:.2f}, {partition[i+1]:.2f})")
    return intervals

# ---------------------------- Main Code ---------------------------- #


args = ParseArguments()
n=int(args.n)
results_path = args.results_path
os.makedirs(results_path, exist_ok=True)

# Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
seed = None if args.seed.lower() == "none" else int(args.seed)
rng = np.random.default_rng(np.random.PCG64(seed=seed))


# Define partition: note that partition is an array with 6 values,
# so the bins are 5 intervals.
partition = np.array([0, 0.15, 0.35, 0.6, 0.8, 1])

# --- Display Partition Information ---
print("\n--- Partition Information ---")
intervals = format_partition(partition)
for inter in intervals:
    print(inter)

# Compute bin probabilities as differences of the partition
bin_probs = np.diff(partition)  # shape (5,)
print("\nBin probabilities (lengths of each interval):", np.round(bin_probs, 3))

# ------------------- Generate the three sets ------------------- #

# Set A: standard uniform random numbers

data_A = rng.random(size=n)


# Set B: transformation of set A using exponential transform and rescaling
exp_data = np.exp(data_A - 1)
min_exp = np.min(exp_data) - 1e-7
max_exp = np.max(exp_data) + 1e-7
data_B = (exp_data - min_exp) / (max_exp - min_exp)

# Set C: quasi-random numbers from Sobol generator
#sampler = qmc.Sobol(d=1, scramble=False, seed=7)
halton_sampler = qmc.Halton(d=1, scramble=False, seed=31415)
data_C = halton_sampler .random(n).flatten()

print("\nMean of set A:", np.mean(data_A))
print("Mean of set B:", np.mean(data_B))
print("Mean of set C:", np.mean(data_C))


# ------------------- Kolmogorov-Smirnov Test ------------------- #
print("\n############################")
print("## Kolmogorov-Smirnov Test   ##")
print("############################\n")
D_A = compute_KS_statistic(data_A, n)
D_B = compute_KS_statistic(data_B, n)
D_C = compute_KS_statistic(data_C, n)

pval_A_KS = ks_pvalue(D_A, n)
pval_B_KS = ks_pvalue(D_B, n)
pval_C_KS = ks_pvalue(D_C, n)



print(f"D_n^A(obs) = {D_A:.5f}   --> p-value = {pval_A_KS*100:.2f}%")
print(f"D_n^B(obs) = {D_B:.5f}   --> p-value = {pval_B_KS*100:.2f}%")
print(f"D_n^C(obs) = {D_C:.5f}   --> p-value = {pval_C_KS*100:.2f}%")

# ------------------- Chi-square Test ------------------- #
print("\n############################")
print("## Chi-square Test Details  ##")
print("##############################\n")

# Observed counts in each bin for each set
counts_A = compute_counts(partition, data_A)
counts_B = compute_counts(partition, data_B)
counts_C = compute_counts(partition, data_C)
total_C = np.sum(counts_C)
# For set C, adjust expected counts based on observed total:
expected_C = total_C * bin_probs / np.sum(bin_probs)
expected_all = n * bin_probs  # for sets A and B

# Compute chi-square contributions for each bin
chi2_A_contrib = (counts_A - expected_all)**2 / expected_all
chi2_B_contrib = (counts_B - expected_all)**2 / expected_all
chi2_C_contrib = (counts_C - expected_C)**2 / expected_C

# Build a table for display (mimicking the book's table)
table_data = []
for i in range(len(counts_A)):
    # Interval: first bin is (0, a), subsequent are [a,b)
    interval_str = f"(0, {partition[i+1]:.2f})" if i == 0 else f"[{partition[i]:.2f}, {partition[i+1]:.2f})"
    row = [
        interval_str,
        f"{expected_all[i]:.2f}",
        counts_A[i],
        f"{chi2_A_contrib[i]:.2f}",
        counts_B[i],
        f"{chi2_B_contrib[i]:.2f}",
        counts_C[i],
        f"{chi2_C_contrib[i]:.2f}"
    ]
    table_data.append(row)

headers = ["Interval", "Exp.", "O_A", "χ²_A_interv", "O_B", "χ²_B_interv", "O_C", "χ²_C_interv"]



print(tabulate(table_data, headers=headers, tablefmt="grid"))

chi2_A_total = np.sum(chi2_A_contrib)
chi2_B_total = np.sum(chi2_B_contrib)
chi2_C_total = np.sum(chi2_C_contrib)

pval_A_chi2 = chisquare(counts_A, expected_all)[1]
pval_B_chi2 = chisquare(counts_B, expected_all)[1]
pval_C_chi2 = chisquare(counts_C, expected_C)[1]

print("\nOverall Chi-square statistics:")
print(f"Set A: χ² = {chi2_A_total:.2f}   --> p-value = {pval_A_chi2*100:.2f}%")
print(f"Set B: χ² = {chi2_B_total:.2f}   --> p-value = {pval_B_chi2*100:.2f}%")
print(f"Set C: χ² = {chi2_C_total:.2f}   --> p-value = {pval_C_chi2*100:.2f}%")

# ------------------- Plots ------------------- #
# Plot the three sets with partition shading

fig, ax = plt.subplots(figsize=(9, 2))

ax.set_yticklabels([])

patches = []
# Use the intervals defined by partition
colors = ['#cbcbcb', '#ebebeb', '#cbcbcb', '#ebebeb', '#cbcbcb']
for i in range(len(partition)-1):
    rect = matplotlib.patches.Rectangle((partition[i], -0.1), partition[i+1]-partition[i], 2.2)
    patches.append(rect)
patch_collection = matplotlib.collections.PatchCollection(patches, cmap=matplotlib.colors.ListedColormap(colors))
patch_collection.set_array(np.arange(len(patches)))
ax.add_collection(patch_collection)

plt.scatter(data_A,   np.zeros(n), color='blue', marker='o',  facecolors='none', edgecolors='blue', s=50, label="Set A")
plt.scatter(data_B,   np.ones(n),  color='red',   marker='x',    s=50, label="Set B")
plt.scatter(data_C, 2*np.ones(n), color='green', marker='^', facecolors='none', edgecolors='green', s=50, label="Set C")
plt.title(f"Three sets of n={n} points from [0,1): A (blue), B (red), C (green)")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(results_path, "ch2_prng_sets_A_B_C_points_plot.pdf"), bbox_inches='tight')


# Plot ECDFs
x_A, y_A = ecdf(data_A)
x_B, y_B = ecdf(data_B)
x_C, y_C = ecdf(data_C)
x0 = np.linspace(0, 1, 200)

plt.figure(figsize=(4,4))
plt.step(x_A, y_A, label="Empirical c.d.f Set A", color='blue')
plt.plot(x0, x0, 'k--', label="c.d.f of Uniform(0,1)")
plt.title("Empirical c.d.f of Set A")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_path, "ch2_prng_sets_A_B_C_empCDF_of_setA.pdf"), bbox_inches='tight')

plt.figure(figsize=(4,4))
plt.step(x_B, y_B, label="Empirical c.d.f. Set B", color='red')
plt.plot(x0, x0, 'k--', label="c.d.f. of Uniform(0,1)")
plt.title("Empirical c.d.f of Set B")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_path, "ch2_prng_sets_A_B_C_empCDF_of_setB.pdf"), bbox_inches='tight')


plt.figure(figsize=(4,4))
plt.step(x_C, y_C, label="Empirical c.d.f Set C", color='green')
plt.plot(x0, x0, 'k--', label="c.d.f. of Uniform(0,1)")
plt.title("Empirical c.d.f. of Set C")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_path, "ch2_prng_sets_A_B_C_empCDF_of_setC.pdf"), bbox_inches='tight')


# ------------------- Frequency of Pairs Test (m = 2) ------------------- #
print("\n############################")
print("## Frequency of Pairs Test  ##")
print("##############################\n")

# We assume n is even; here r = n//2 pairs.
L = 3  # number of intervals per side; hence k = L^2 boxes.
r = n // 2
expected_pair = r / (L*L)  # expected count per box

def frequency_pairs_counts(data, L):
    """
    Given a sequence 'data' (of even length) in [0,1),
    group the numbers into pairs and compute a frequency table for the pair (floor(L*u1), floor(L*u2)).
    Returns an array of shape (L, L) with counts.
    """
    data = np.array(data)  # ensure array
    r = len(data) // 2
    # Transform data into integer labels in {0, ..., L-1}
    labels = np.floor(L * data).astype(int)
    counts = np.zeros((L, L), dtype=int)
    for j in range(r):
        s = labels[2*j]
        t = labels[2*j+1]
        counts[s, t] += 1
    return counts

# Compute frequency pairs counts for each set
freq_A = frequency_pairs_counts(data_A, L)
freq_B = frequency_pairs_counts(data_B, L)
freq_C = frequency_pairs_counts(data_C, L)

tmp=rng.random(size=1)

# Also compute a shuffled version of set C (denoted as C')
data_C_shuffled = cp.copy(data_C)
rng.shuffle(data_C_shuffled)
freq_C_shuffled = frequency_pairs_counts(data_C_shuffled, L)

# Compute chi-square statistics for each frequency table.
def chi2_from_freq(freq, expected):
    # Flatten the frequency table.
    freq_flat = freq.flatten()
    f_exp = np.full_like(freq_flat, expected, dtype=float)
    chi2_stat, p_val = chisquare(freq_flat, f_exp)
    return chi2_stat, p_val

chi2_A_pairs, pval_A_pairs = chi2_from_freq(freq_A, expected_pair)
chi2_B_pairs, pval_B_pairs = chi2_from_freq(freq_B, expected_pair)
chi2_C_pairs, pval_C_pairs = chi2_from_freq(freq_C, expected_pair)
chi2_Cs_pairs, pval_Cs_pairs = chi2_from_freq(freq_C_shuffled, expected_pair)

# Prepare tables for display using tabulate.
def table_from_freq(freq):
    # Create table rows for each row of the grid.
    table = []
    for row in freq:
        table.append([str(x) for x in row])
    return table




print(f"Using L = {L}, r = {r} pairs, expected count per box = {expected_pair:.2f}\n")

print("Observed frequency table for Set A:")
print(tabulate(table_from_freq(freq_A), tablefmt="grid"))
print(f"Chi-square: {chi2_A_pairs:.2f}   --> p-value: {pval_A_pairs*100:.2f}%\n")

print("Observed frequency table for Set B:")
print(tabulate(table_from_freq(freq_B), tablefmt="grid"))
print(f"Chi-square: {chi2_B_pairs:.2f}   --> p-value: {pval_B_pairs*100:.2f}%\n")

print("Observed frequency table for Set C:")
print(tabulate(table_from_freq(freq_C), tablefmt="grid"))
print(f"Chi-square: {chi2_C_pairs:.2f}   --> p-value: {pval_C_pairs*100:.2f}%\n")

print("Observed frequency table for Shuffled Set C (C'):")
print(tabulate(table_from_freq(freq_C_shuffled), tablefmt="grid"))
print(f"Chi-square: {chi2_Cs_pairs:.2f}   --> p-value: {pval_Cs_pairs*100:.2f}%\n")

# ------------------- Second Level Testing ------------------- #
print("\n############################")
print("## Second Level Testing   ##")
print("############################\n")

R = 500  # number of sequences; adjust as needed

# Arrays to store p-values for each sequence (for both tests)
pvals_KS_A = np.zeros(R)
pvals_KS_B = np.zeros(R)
pvals_KS_C = np.zeros(R)
pvals_chi2_A = np.zeros(R)
pvals_chi2_B = np.zeros(R)
pvals_chi2_C = np.zeros(R)

# For each sequence i = 0,...,R-1:
for i in range(R):
    # ---------- Set A ----------
    rng_i = default_rng(PCG64(seed=i))
    A_i = rng_i.random(size=n)
    counts_A_i = compute_counts(partition, A_i)
    # For sets A and B, expected counts are n * bin_probs
    expected_A = n * bin_probs
    _, pval_chi2_A_i = chisquare(counts_A_i, expected_A)
    pvals_chi2_A[i] = pval_chi2_A_i
    pvals_KS_A[i] = kstest(A_i, cdf='uniform')[1]

    # ---------- Set B ----------
    exp_data_i = np.exp(A_i - 1)
    # Use epsilon = 1e-17 as in the text:
    min_exp_i = np.min(exp_data_i) - 1e-17
    max_exp_i = np.max(exp_data_i) + 1e-17
    B_i = (exp_data_i - min_exp_i) / (max_exp_i - min_exp_i)
    counts_B_i = compute_counts(partition, B_i)
    total_B_i = np.sum(counts_B_i)
    if total_B_i != n:
        expected_B_i = total_B_i * bin_probs / np.sum(bin_probs)
    else:
        expected_B_i = n * bin_probs
    _, pval_chi2_B_i = chisquare(counts_B_i, expected_B_i)
    pvals_chi2_B[i] = pval_chi2_B_i
    pvals_KS_B[i] = kstest(B_i, cdf='uniform')[1]

    # ---------- Set C ----------
    # Use Halton sampler with seed=i
    halton_i = qmc.Halton(d=1, scramble=False, seed=i)
    C_i = halton_i.random(n).flatten()
    counts_C_i = compute_counts(partition, C_i)
    total_C_i = np.sum(counts_C_i)
    expected_C_i = total_C_i * bin_probs / np.sum(bin_probs)
    _, pval_chi2_C_i = chisquare(counts_C_i, expected_C_i)
    pvals_chi2_C[i] = pval_chi2_C_i
    pvals_KS_C[i] = kstest(C_i, cdf='uniform')[1]

# ---------- Bin the p-values (final level) ----------
# Define final partition for p-values: 10 equal intervals in [0,1]
partition_final = np.arange(0, 1.1, 0.1)


def bin_pvalues(pvals, partition):
    """Bin the p-values according to the given partition."""
    counts = np.zeros(len(partition) - 1, dtype=int)
    for j in range(len(counts)):
        lower = partition[j]
        upper = partition[j + 1]
        if j == 0:
            counts[j] = np.sum((pvals > lower) & (pvals <= upper))
        else:
            counts[j] = np.sum((pvals >= lower) & (pvals < upper))
    return counts


bins_final_A_chi2 = bin_pvalues(pvals_chi2_A, partition_final)
bins_final_B_chi2 = bin_pvalues(pvals_chi2_B, partition_final)
bins_final_C_chi2 = bin_pvalues(pvals_chi2_C, partition_final)

bins_final_A_KS = bin_pvalues(pvals_KS_A, partition_final)
bins_final_B_KS = bin_pvalues(pvals_KS_B, partition_final)
bins_final_C_KS = bin_pvalues(pvals_KS_C, partition_final)

# Expected number per bin is R*(length of interval); here each is R*0.1
expected_final = R * np.diff(partition_final)

# Final chi-square tests on the p-value distributions:
chi2_final_A_chi2, p_final_A_chi2 = chisquare(bins_final_A_chi2, expected_final)
chi2_final_B_chi2, p_final_B_chi2 = chisquare(bins_final_B_chi2, expected_final)
chi2_final_C_chi2, p_final_C_chi2 = chisquare(bins_final_C_chi2, expected_final)

chi2_final_A_KS, p_final_A_KS = chisquare(bins_final_A_KS, expected_final)
chi2_final_B_KS, p_final_B_KS = chisquare(bins_final_B_KS, expected_final)
chi2_final_C_KS, p_final_C_KS = chisquare(bins_final_C_KS, expected_final)


# ---------- Plotting the Distribution of p-values ----------
# We'll produce two plots: one for chi-square test p-values and one for KS test p-values.
# Use a background that shows the final partition (10 intervals).

def plot_pvalues(pvals_A, pvals_B, pvals_C, test_label, filename):


    fig, ax = plt.subplots(figsize=(9, 2))
    ax.set_yticklabels([])

    # Define final partition for p-values: 10 equal intervals in [0,1]
    partition_final = np.arange(0, 1.1, 0.1)  # will be [0,0.1,0.2,...,1.0]

    # Create a background with rectangles of width 0.1 and alternating colors.
    bg_colors = ['#cbcbcb', '#ebebeb']
    for j in range(len(partition_final) - 1):
        rect = matplotlib.patches.Rectangle(
            (partition_final[j], -0.1),  # bottom-left corner
            partition_final[j + 1] - partition_final[j],  # width = 0.1
            2.2,  # height
            facecolor=bg_colors[j % 2],
            edgecolor='none'
        )
        ax.add_patch(rect)

    # Determine R from the length of pvals_A
    R = len(pvals_A)



    # Plot the p-values for each procedure at different y-levels.
    ax.scatter(pvals_A, np.zeros(R), marker='o', facecolors='none', edgecolors='blue', s=50)
    ax.scatter(pvals_B, np.ones(R), color='red', marker='x', s=50)
    ax.scatter(pvals_C, 2 * np.ones(R), color='green', marker='^', facecolors='none', edgecolors='green', s=50)

    ax.set_xlabel("p-value")
    ax.set_title(f"Second Level Testing: {test_label}")
    #ax.legend()

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')


# ------------------- Display p-value Frequency Tables ------------------- #
# Define header for 10 bins: P1, P2, ..., P10
headers = ["", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]

# Build table for chi-square test p-values
table_chi2 = [
    ["A"] + list(bins_final_A_chi2),
    ["B"] + list(bins_final_B_chi2),
    ["C"] + list(bins_final_C_chi2)
]
print("\nFrequency table for chi-square test p-values:")
print(tabulate(table_chi2, headers=headers, tablefmt="grid"))

# Build table for KS test p-values
table_KS = [
    ["A"] + list(bins_final_A_KS),
    ["B"] + list(bins_final_B_KS),
    ["C"] + list(bins_final_C_KS)
]
print("\nFrequency table for KS test p-values:")
print(tabulate(table_KS, headers=headers, tablefmt="grid"))

# ---------- Print Final Results ----------
print("\n--- Second Level Testing: Final p-values (Chi-square Test on p-values) ---")
print(f"Set A: chi²_final = {chi2_final_A_chi2:.4f}, p_final = {p_final_A_chi2:.4f}")
print(f"Set B: chi²_final = {chi2_final_B_chi2:.4f}, p_final = {p_final_B_chi2:.4e}")
print(f"Set C: chi²_final = {chi2_final_C_chi2:.4f}, p_final = {p_final_C_chi2:.4e}")

print("\n--- Second Level Testing: Final p-values (KS Test on p-values) ---")
print(f"Set A: chi²_final = {chi2_final_A_KS:.4f}, p_final = {p_final_A_KS:.4f}")
print(f"Set B: chi²_final = {chi2_final_B_KS:.4f}, p_final = {p_final_B_KS:.4e}")
print(f"Set C: chi²_final = {chi2_final_C_KS:.4f}, p_final = {p_final_C_KS:.4e}")



# ------------------- Final Comparison Table ------------------- #
# Prepare data for final comparison table.

# Format the final chi-square statistics and p-values as strings.
# Adjust the formatting as needed.
row1 = ["χ² Test", "χ²_final(obs)", f"{chi2_final_A_chi2:.3f}", f"{chi2_final_B_chi2:.3f}", f"{chi2_final_C_chi2:.3f}"]
row2 = ["", "p_final", f"{p_final_A_chi2:.3g}", f"{p_final_B_chi2:.3g}", f"{p_final_C_chi2:.3g}"]
row3 = ["KS Test", "χ²_final(obs)", f"{chi2_final_A_KS:.3f}", f"{chi2_final_B_KS:.3f}", f"{chi2_final_C_KS:.3f}"]
row4 = ["", "p_final", f"{p_final_A_KS:.3g}", f"{p_final_B_KS:.3g}", f"{p_final_C_KS:.3g}"]

# Combine rows into a table.
table_final = [row1, row2, row3, row4]

# Create headers for the table: first column is empty, then A, B, C.
headers_final = ["", "", "A", "B", "C"]

print("\n--- Final Comparison Table ---")
print(tabulate(table_final, headers=headers_final, tablefmt="grid"))


plot_pvalues(pvals_chi2_A, pvals_chi2_B, pvals_chi2_C,
             "R=500 p-values from Chi-square Test", os.path.join(results_path,"ch2_prng_sets_A_B_C_second_level_chi2_pvalues.pdf"))
plot_pvalues(pvals_KS_A, pvals_KS_B, pvals_KS_C,
             "R=500 p-values from KS Test", os.path.join(results_path,"ch2_prng_sets_A_B_C_second_level_KS_pvalues.pdf"))


plt.show()