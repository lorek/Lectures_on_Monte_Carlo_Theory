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
## File: ch1_2d_game_winning_prob.py                                ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import qmc




def ParseArguments():
    parser = argparse.ArgumentParser(description="Estimating winning probability in 2D game")
    parser.add_argument('--R', default="5000", required=False,
                        help='Total number of games (default: %(default)s)')
    parser.add_argument('--R0', default="400", required=False,
                        help='Starting game index (default: %(default)s)')
    parser.add_argument('--x_start', default="3", required=False,
                        help='Starting x position (default: %(default)s)')
    parser.add_argument('--y_start', default="2", required=False,
                        help='Starting y position (default: %(default)s)')
    parser.add_argument('--results_path', default="results", help='Path for saving results (default: %(default)s)')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")
    args = parser.parse_args()
    return args


def compute_xy(p_a, q_a, p_b, q_b, x, y):
    """
    Compute a product-of-sums value used in the winning probability formula.
    p_a, q_a, p_b, q_b: 1D arrays.
    x, y: number of steps in each dimension.
    """
    ratio_a = (q_a / p_a).reshape(-1)
    ratio_b = (q_b / p_b).reshape(-1)
    sum1 = 0
    for nj in np.arange(x):
        prod_val = np.prod(ratio_a[:nj])
        sum1 += prod_val
    sum2 = 0
    for nj in np.arange(y):
        prod_val = np.prod(ratio_b[:nj])
        sum2 += prod_val
    return sum1 * sum2


def compute_counts(partition, data):
    """
    Compute counts of data in each bin defined by partition.
    The first bin is (0, partition[1]) and subsequent bins are [partition[i], partition[i+1]).
    """
    counts = np.zeros(len(partition) - 1, dtype=int)
    for i in range(len(counts)):
        lower = partition[i]
        upper = partition[i + 1]
        if i == 0:
            counts[i] = np.sum((data > lower) & (data <= upper))
        else:
            counts[i] = np.sum((data >= lower) & (data < upper))
    return counts


######################################
#            Main Code               #
######################################

# Parse command-line arguments

args = ParseArguments()
R_games = int(args.R)
R0 = int(args.R0)
results_path = args.results_path
os.makedirs(results_path, exist_ok=True)


x_start = int(args.x_start) #3
y_start = int(args.y_start) #2


# Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
seed = None if args.seed.lower() == "none" else int(args.seed)
rng = np.random.default_rng(np.random.PCG64(seed=seed))
np.random.seed(31415)

# ---------------------------------------------------------------------
# Preliminary simulation: compute winning probability in a 2D game.
# For a symmetric example, let N1 = N2 = 4 and for i=1,2,3, p_j(i)=q_j(i)=1/4.
# For initial assets (2,3), the formula gives rho((2,3)) = 3/8 = 0.375.
######################################

p_a = np.array([0, 1 / 4, 1 / 4, 1 / 4, 0])
q_a = np.array([0, 1 / 4, 1 / 4, 1 / 4, 0])
p_b = np.array([0, 1 / 4, 1 / 4, 1 / 4, 0])
q_b = np.array([0, 1 / 4, 1 / 4, 1 / 4, 0])

n = 3

rho_reference = compute_xy(p_a[1:n+1], q_a[1:n+1], p_b[1:n+1], q_b[1:n+1], 2, 3) / \
                compute_xy(p_a[1:n+1], q_a[1:n+1], p_b[1:n+1], q_b[1:n+1], 4, 4)


print("Reference rho((2,3)) =", rho_reference)


# ---------------------------------------------------------------------
# Simulation of games.
######################################




actions = ['L', 'R', 'U', 'D', 'S']

# Initialize results storage.
results = np.zeros((4, R_games))
# Also store 2D random pairs for plotting.
pairs_PCG64 = []  # PCG64 (MT)
pairs_LH = []  # LatinHypercube
pairs_Sobol = []  # Sobol
pairs_Halton = []  # Halton

# Create QMC generators with seed=31415.
halton_model = qmc.Halton(d=2, scramble=True, seed=31415)
lh_model = qmc.LatinHypercube(d=2, seed=31415)
sobol_model = qmc.Sobol(d=2, scramble=True, seed=31415)

# We now generate samples on demand (using .random(1) each time).



print(f"\nNow performing {R_games} games using various generators...\n")

rho_estimates = np.zeros(4)
total_steps = np.zeros(4)
generator_names = ["PCG64", "LatinHypercube", "Sobol", "Halton"]

# Main simulation loop.
for gen_idx in np.arange(4):
    win_count = 0
    loss_count = 0
    steps = 0
    print(f"Processing generator {generator_names[gen_idx]} ...")
    for game in np.arange(R_games):

        # Initialize game state.
        x = x_start
        y = y_start
        continue_game = True
        while continue_game:
            # Ensure state stays in valid bounds [0, 4].
            x = max(0, min(x, 4))
            y = max(0, min(y, 4))
            steps += 1
            x_old, y_old = x, y


            # Draw a 2D random pair on demand.
            if gen_idx == 0:
                U1, U2  = rng.random(2)
                pairs_PCG64.append([U1, U2])
            elif gen_idx == 1:
                U1, U2 = lh_model.random(1).flatten()
                pairs_LH.append([U1, U2])
            elif gen_idx == 2:
                U_pair = sobol_model.random(1).flatten() + 1e-1 * np.random.rand(2)
                U1, U2 = U_pair
                U1 = U1 if U1 < 1 else U1 - 1
                U2 = U2 if U2 < 1 else U2 - 1

                U1 = U1 if U1 > 0 else U1 + 1
                U2 = U2 if U2 > 0 else U2 + 1


                pairs_Sobol.append([U1, U2])
            elif gen_idx == 3:
                U1, U2 = halton_model.random(1).flatten()
                pairs_Halton.append([U1, U2])


            # Decide move using U1, U2.
            if U1 < p_a[x] + q_a[x]:
                if U2 < q_a[x] / (p_a[x] + q_a[x]):
                    move = 'L'
                else:
                    move = 'R'
            elif (U1 >= p_a[x] + q_a[x]) and (U1 <= p_a[x] + q_a[x] + p_b[y] + q_b[y]):
                if U2 < q_b[y] / (p_b[y] + q_b[y]):
                    move = 'D'
                else:
                    move = 'U'
            elif U1 >= p_a[x] + q_a[x] + p_b[y] + q_b[y]:
                move = 'S'
            # Update state based on move.
            if move == 'L':
                x -= 1
            elif move == 'R':
                x += 1
            elif move == 'U':
                y += 1
            elif move == 'D':
                y -= 1

            # Check termination conditions.
            if (x == 0 or y == 0):
                loss_count += 1
                continue_game = False
            if (x == 4 and y == 4):
                win_count += 1
                results[gen_idx, game] = 1
                continue_game = False
        # End of a single game.
    rho_estimates[gen_idx] = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else np.nan
    total_steps[gen_idx] = steps / R_games
    print(f"Estimated winning probability for {generator_names[gen_idx]}: {rho_estimates[gen_idx]:.4f}")




table_data = [
    ["Estimated prob", f"{rho_estimates[0]:.4f}", f"{rho_estimates[1]:.4f}", f"{rho_estimates[2]:.4f}", f"{rho_estimates[3]:.4f}"],
    ["Avg nr of steps", f"{total_steps[0]:.4f}", f"{total_steps[1]:.4f}", f"{total_steps[2]:.4f}", f"{total_steps[3]:.4f}"]
]
headers = ["", "PCG64", "LatinHypercube", "Sobol", "Halton"]

print("\nWinning probabilities and average steps per game:")
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Compute cumulative win estimates.
win_est_PCG64 = np.cumsum(results[0, :]) / np.arange(1, R_games + 1)
win_est_LH = np.cumsum(results[1, :]) / np.arange(1, R_games + 1)
win_est_Sobol = np.cumsum(results[2, :]) / np.arange(1, R_games + 1)
win_est_Halton = np.cumsum(results[3, :]) / np.arange(1, R_games + 1)
games_axis = np.arange(1, R_games + 1)

# First plot: games 0 to R0-1
fig1, ax1 = plt.subplots(figsize=(2, 4))
ax1.plot(games_axis[:R0], win_est_PCG64[:R0], label="PCG64", color='b', alpha=1)
ax1.plot(games_axis[:R0], win_est_LH[:R0], label="LatinHypercube", color='g', alpha=1)
ax1.plot(games_axis[:R0], win_est_Sobol[:R0], label="Sobol", color='k', alpha=1)
ax1.plot(games_axis[:R0], win_est_Halton[:R0], label="Halton", color='c', alpha=1)
ax1.axhline(y=rho_reference, color='red',    label='True $\\rho$')
#ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_path, "ch1_2d_game_win_prob_0_to_R0.pdf"), bbox_inches='tight')
ax1.set_title("Winning Probability Estimates (Games 0 to R0-1)")

# Second plot: games R0 to R_games-1
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(games_axis[R0:], win_est_PCG64[R0:], label="PCG64", color='b', alpha=1)
ax2.plot(games_axis[R0:], win_est_LH[R0:], label="LatinHypercube", color='g', alpha=1)
ax2.plot(games_axis[R0:], win_est_Sobol[R0:], label="Sobol", color='k', alpha=1)
ax2.plot(games_axis[R0:], win_est_Halton[R0:], label="Halton", color='c', alpha=1)
ax2.axhline(y=rho_reference, color='red',   label='True $\\rho$')
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_path, "ch1_2d_game_win_prob_R0_to_R.pdf"), bbox_inches='tight')
ax2.set_title("Winning Probability Estimates (Games R0 to R-1)")

# ---------------------------------------------------------------------
# Plotting the 2D points used in the simulations.

num_initial = 15
num_all = 1000


def plot_points(points_array, title, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(points_array[:num_all, 0], points_array[:num_all, 1], s=2, color='gray', alpha=0.6)
    for idx in range(num_initial):
        plt.scatter(points_array[idx, 0], points_array[idx, 1], s=50, color='black', marker='+')
        plt.text(points_array[idx, 0], points_array[idx, 1], str(idx), color='red', fontsize=12)
    plt.savefig(filename, bbox_inches='tight')
    plt.title(title)



pairs_PCG64_arr = np.array(pairs_PCG64)
pairs_LH_arr = np.array(pairs_LH)
pairs_Sobol_arr = np.array(pairs_Sobol)
pairs_Halton_arr = np.array(pairs_Halton)

plot_points(pairs_PCG64_arr, "PCG64 Points", os.path.join(results_path, "ch1_2d_game_pcg64_points.pdf"))
plot_points(pairs_LH_arr, "LatinHypercube Points", os.path.join(results_path, "ch1_2d_game_lh_points.pdf"))
plot_points(pairs_Sobol_arr, "Sobol Points", os.path.join(results_path, "ch1_2d_game_sobol_points.pdf"))
plot_points(pairs_Halton_arr, "Halton Points", os.path.join(results_path, "ch1_2d_game_halton_points.pdf"))


# ---------------------------------------------------------------------
# Plot histograms of the sums U1+U2 for each generator.
def plot_histogram(data_sum, title, filename):
    bins_number = 40
    xx1 = np.linspace(0, 1, 100)
    yy1 = xx1
    xx2 = np.linspace(1, 2, 100)
    yy2 = 2 - xx2
    plt.figure(figsize=(6, 5))
    plt.hist(data_sum, bins_number, density=True, facecolor='black', alpha=0.5, rwidth=0.8)
    plt.plot(xx1, yy1, color='b')
    plt.plot(xx2, yy2, color='b')
    plt.savefig(filename, bbox_inches='tight')
    plt.title(title)



MT_sum = pairs_PCG64_arr[:, 0] + pairs_PCG64_arr[:, 1]
LH_sum = pairs_LH_arr[:, 0] + pairs_LH_arr[:, 1]
Sobol_sum = pairs_Sobol_arr[:, 0] + pairs_Sobol_arr[:, 1] if pairs_Sobol_arr.size > 0 else np.array([])
Halton_sum = pairs_Halton_arr[:, 0] + pairs_Halton_arr[:, 1]

plot_histogram(MT_sum, "PCG64 Sum Histogram", os.path.join(results_path,"ch1_2d_game_pcg64_hist.pdf"))
plot_histogram(LH_sum, "LatinHypercube Sum Histogram", os.path.join(results_path,"ch1_2d_game_lh_hist.pdf"))
if Sobol_sum.size > 0:
    plot_histogram(Sobol_sum, "Sobol Sum Histogram", os.path.join(results_path,"ch1_2d_game_sobol_hist.pdf"))
plot_histogram(Halton_sum, "Halton Sum Histogram", os.path.join(results_path,"ch1_2d_game_halton_hist.pdf"))

plt.show()
