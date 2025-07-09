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
## File: ch5_scheduling_means.py                                    ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import argparse
from tabulate import tabulate
from numpy.random import default_rng, PCG64

def ParseArguments():
    """
    Parse command-line arguments.
    The --R argument accepts a comma‐separated list of replication counts.
    """
    parser = argparse.ArgumentParser(description='Scheduling simulation: CMC and antithetic.')
    parser.add_argument('--N', type=int, default=10,
                        help='Number of tasks to schedule, default: %(default)s')
    parser.add_argument('--R', type=str, default='1000,100000',
                        help='Comma-separated list of replication counts (e.g., "1000,100000")')
    parser.add_argument('--seed', default="31415",
                        help="Seed for the PRNG. Use 'None'  for no fixed seed. (default: %(default)s)")

    args = parser.parse_args()

    return args




def simulate_FCFS(R, N, rng):
    """
    Crude Monte Carlo simulation for the FCFS scheduling rule.
    (See FCFS.m :contentReference[oaicite:0]{index=0})
    """
    T_list = []
    for _ in range(R):
        current_N = N
        X = -np.log(rng.uniform(0,1,N))
        Y = list(X)
        T_val = 0.0
        while current_N > 1:
            T_val += min(Y[0], Y[1])
            Y[0] = abs(Y[1] - Y[0])
            Y.pop(1)
            current_N -= 1
        T_val += Y[0]
        T_list.append(T_val)
    T_array = np.array(T_list)
    mean_val = np.mean(T_array)
    std_val = np.std(T_array, ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf


def simulate_FCFS_antithetic(R, N, rng):
    """
    Antithetic simulation for FCFS.
    (See FCFSanty.m :contentReference[oaicite:1]{index=1})
    """
    C_list = []
    for _ in range(R // 2):
        current_N = N
        u = rng.uniform(0,1,N)
        X1 = -np.log(u)
        X2 = -np.log(1 - u)
        T1 = 0.0
        Y = list(X1)
        cm = current_N
        while cm > 1:
            T1 += min(Y[0], Y[1])
            Y[0] = abs(Y[1] - Y[0])
            Y.pop(1)
            cm -= 1
        T1 += Y[0]
        T2 = 0.0
        Z = list(X2)
        cm = current_N
        while cm > 1:
            T2 += min(Z[0], Z[1])
            Z[0] = abs(Z[1] - Z[0])
            Z.pop(1)
            cm -= 1
        T2 += Z[0]
        C_list.append((T1 + T2) / 2)
    C_array = np.array(C_list)
    mean_val = np.mean(C_array)
    std_val = np.std(C_array, ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf


def simulate_LRPT(R, N, rng):
    """
    Crude Monte Carlo simulation for the LRPT scheduling rule.
    (See LPTF.m :contentReference[oaicite:2]{index=2})
    """
    T_list = []
    for _ in range(R):
        current_N = N
        X = -np.log(rng.uniform(0,1,N))
        Y = sorted(X)
        T_val = 0.0
        while current_N > 1:
            t_add = min(Y[-1], Y[-2])
            T_val += t_add
            new_val = abs(Y[-1] - Y[-2])
            Y.pop()
            Y[-1] = new_val
            current_N -= 1
        T_val += Y[0]
        T_list.append(T_val)
    T_array = np.array(T_list)
    mean_val = np.mean(T_array)
    std_val = np.std(T_array, ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf


def simulate_LRPT_antithetic(R, N, rng):
    """
    Antithetic simulation for LRPT.
    (See LPTFanty.m :contentReference[oaicite:3]{index=3})
    """
    C_list = []
    for _ in range(R // 2):
        current_N = N
        u = rng.uniform(0,1,N)
        X1 = -np.log(u)
        X2 = -np.log(1 - u)
        Y = sorted(X1)
        Z = sorted(X2)
        T1 = 0.0
        T2 = 0.0
        cm = current_N
        while cm > 1:
            T1 += min(Y[-1], Y[-2])
            new_val = abs(Y[-1] - Y[-2])
            Y.pop()
            Y[-1] = new_val
            T2 += min(Z[-1], Z[-2])
            new_val2 = abs(Z[-1] - Z[-2])
            Z.pop()
            Z[-1] = new_val2
            cm -= 1
        T1 += Y[0]
        T2 += Z[0]
        C_list.append((T1 + T2) / 2)
    C_array = np.array(C_list)
    mean_val = np.mean(C_array)
    std_val = np.std(C_array, ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf


def simulate_SRPT(R, N, rng):
    """
    Symulacja SRPT niemal 1:1 z kodem MATLAB-owym:
      - W każdej z R replik generujemy 1 zestaw N losowych czasów Exp(1).
      - Sortujemy je (Y).
      - W pętli sumujemy min(Y[0], Y[1]) i aktualizujemy listę
      - Wynik T wrzucamy do T_list.
    Zwraca (mean_val, std_val, half_conf).
    """
    T_list = []
    for _ in range(R):
        # Losowanie N czasów Exp(1)
        X = -np.log(rng.uniform(size=N))
        # Sortowanie
        Y = list(np.sort(X))
        m = N
        T_val = 0.0

        # Pętla "while (m>1)"
        while m > 1:
            T_val += min(Y[0], Y[1])
            Y[0] = abs(Y[1] - Y[0])
            Y.pop(1)
            m -= 1
        # Dodajemy ostatni czas
        T_val += Y[0]
        T_list.append(T_val)

    T_array = np.array(T_list)
    mean_val = np.mean(T_array)
    std_val = np.std(T_array, ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf



def simulate_SRPT_antithetic(R, N, rng):
    """
    Antithetic simulation for SRPT.
    (See SPTFanty.m :contentReference[oaicite:5]{index=5})
    """
    C_list = []
    for _ in range(R // 2):
        current_N = N
        u = rng.uniform(0,1,N)
        X1 = -np.log(u)
        X2 = -np.log(1 - u)
        Y = sorted(X1)
        Z = sorted(X2)
        T1 = 0.0
        T2 = 0.0
        cm = current_N
        while cm > 1:
            T1 += min(Y[0], Y[1])
            Y[0] = abs(Y[1] - Y[0])
            Y.pop(1)
            T2 += min(Z[0], Z[1])
            Z[0] = abs(Z[1] - Z[0])
            Z.pop(1)
            cm -= 1
        T1 += Y[0]
        T2 += Z[0]
        C_list.append((T1 + T2) / 2)
    C_array = np.array(C_list)
    mean_val = np.mean(C_array)
    std_val = np.std(C_array, ddof=1)
    half_conf = 1.96 * std_val / np.sqrt(R)
    return mean_val, std_val, half_conf





def main():
    args = ParseArguments()

    # Set seed: if user provided "none" (case-insensitive), seed is set to None (unseeded); otherwise, convert input to int.
    seed = None if args.seed.lower() == "none" else int(args.seed)
    rng = np.random.default_rng(np.random.PCG64(seed=seed))

    N = int(args.N)

    Rs = args.R

    # Convert the comma-separated list of R values into integers
    R_list = [int(r.strip()) for r in Rs.split(',')]


    for R in R_list:
        print(f"\nSimulations with R = {R}")
        # Run Crude Monte Carlo (CMC) simulations
        fcfs_mean, fcfs_std, fcfs_half = simulate_FCFS(R, N, rng)
        srpt_mean, srpt_std, srpt_half = simulate_SRPT(R, N, rng)
        lrpt_mean, lrpt_std, lrpt_half = simulate_LRPT(R, N, rng)

        cmc_table = [
            ["FCFS", f"{fcfs_mean:.4f}", f"{fcfs_std:.4f}", f"{fcfs_half:.4f}"],
            ["SRPT", f"{srpt_mean:.4f}", f"{srpt_std:.4f}", f"{srpt_half:.4f}"],
            ["LRPT", f"{lrpt_mean:.4f}", f"{lrpt_std:.4f}", f"{lrpt_half:.4f}"],
        ]
        print("Crude Monte Carlo (CMC) Results:")
        print(tabulate(cmc_table, headers=["Discipline", "Mean", "Std", "Half CI"], tablefmt="pretty"))

        # Run Antithetic simulations
        fcfs_mean_a, fcfs_std_a, fcfs_half_a = simulate_FCFS_antithetic(R, N, rng)
        srpt_mean_a, srpt_std_a, srpt_half_a = simulate_SRPT_antithetic(R, N, rng)
        lrpt_mean_a, lrpt_std_a, lrpt_half_a = simulate_LRPT_antithetic(R, N, rng)

        anty_table = [
            ["FCFS", f"{fcfs_mean_a:.4f}", f"{fcfs_std_a:.4f}", f"{fcfs_half_a:.4f}"],
            ["SRPT", f"{srpt_mean_a:.4f}", f"{srpt_std_a:.4f}", f"{srpt_half_a:.4f}"],
            ["LRPT", f"{lrpt_mean_a:.4f}", f"{lrpt_std_a:.4f}", f"{lrpt_half_a:.4f}"],
        ]
        print("\nAntithetic Results:")
        print(tabulate(anty_table, headers=["Discipline", "Mean", "Std", "Half CI"], tablefmt="pretty"))


if __name__ == '__main__':
    main()
