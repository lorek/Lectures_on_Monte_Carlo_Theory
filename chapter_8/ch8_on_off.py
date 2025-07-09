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
## File: ch8_on_off.py                                              ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import math
from scipy.stats import pareto, gamma, expon, norm
from tabulate import tabulate


def ParseArguments():
    parser = argparse.ArgumentParser(
        description="On-Off Process with two distribution options (Exp/Exp vs Pareto/Gamma), each producing 2 plots, plus a summary table."
    )
    # General parameters
    parser.add_argument("--results_path", default="results",
                        help="Directory for saving plots (default: %(default)s)")
    parser.add_argument("--seed", default="31415",
                        help="Seed for PCG64 RNG (default: %(default)s)")
    parser.add_argument("--t_max", type=float, default=1000.0,
                        help="Total simulation time (default: %(default)s)")
    parser.add_argument("--t_zoom", type=float, default=100.0,
                        help="Zoomed-in plot range (default: %(default)s)")

    # Exponential On/Off parameters
    parser.add_argument("--exp_on", type=float, default=3.0,
                        help="Mean on-time for exponential in Option1 (default: %(default)s)")
    parser.add_argument("--exp_off", type=float, default=1.0,
                        help="Mean off-time for exponential in Option1 (default: %(default)s)")

    # Pareto On parameters
    parser.add_argument("--pareto_alpha", type=float, default=2.5,
                        help="Pareto alpha shape (default: %(default)s)")
    parser.add_argument("--pareto_scale", type=float, default=1.8,
                        help="Pareto scale xm (default: %(default)s)")

    # Gamma Off parameters
    parser.add_argument("--gamma_shape", type=float, default=2.0,
                        help="Gamma shape (default: %(default)s)")
    parser.add_argument("--gamma_scale", type=float, default=0.5,
                        help="Gamma scale (default: %(default)s)")

    return parser.parse_args()


def simulate_on_off(rng, t_max, sample_on_time, sample_off_time):
    """
    Simulates the on-off process up to t_max, skipping the first cycle,
    returns:
      hatX_t_times, hatX_t_vals  => time-based estimator X̂(t)
      hatI_k_times, hatI_k_vals  => cycle-based estimator I_k
      Z_list, C_list             => on/off durations for the cycles (except first)
      cumul_int                  => integral of X(s) ds over [0, t_max]
    """
    current_time = 0.0
    current_state = 1  # start ON
    first_cycle = True

    Z_list = []  # on-duration of each cycle (except first)
    C_list = []  # total cycle duration (on+off) for each cycle (except first)

    hatX_t_times = []
    hatX_t_vals = []
    hatI_k_times = []
    hatI_k_vals = []

    cumul_int = 0.0  # ∫ X(s) ds

    while current_time < t_max:
        if current_state == 1:
            # ON
            T_on = sample_on_time()
            current_time += T_on
            cumul_int += T_on
            Z_i = T_on
            C_i = T_on
            current_state = 0
        else:
            # OFF
            T_off = sample_off_time()
            current_time += T_off
            C_i += T_off

            if first_cycle:
                first_cycle = False
            else:
                Z_list.append(Z_i)
                C_list.append(C_i)

                sZ = sum(Z_list)
                sC = sum(C_list)
                hatI = sZ / sC
                hatI_k_vals.append(hatI)
                hatI_k_times.append(current_time)

            current_state = 1

        # time-based estimator X̂(t) = ( ∫ X(s) ds ) / t
        # skip t=0 corner case
        if current_time > 0:
            hx = cumul_int / current_time
        else:
            hx = 1.0

        hatX_t_vals.append(hx)
        hatX_t_times.append(current_time)

    # clamp final time if it overshoot
    if hatX_t_times[-1] > t_max:
        hatX_t_times[-1] = t_max

    return (hatX_t_times, hatX_t_vals,
            hatI_k_times, hatI_k_vals,
            Z_list, C_list,
            cumul_int)


def compute_CI(t_max, Z_list, C_list, hatX_final, hatI_final):
    """
    If we have >=2 cycles, compute 95% CI for X(t) and I_k; else None.
    returns (X_low,X_high, I_low,I_high) or None
    """
    k = len(Z_list)
    if k < 2:
        return None

    Z_array = np.array(Z_list)
    C_array = np.array(C_list)
    # zeta_j = Z_j - hatI_final*C_j
    zeta = Z_array - hatI_final*C_array
    zeta_mean = zeta.mean()

    hat_asvarA_sq = np.sum((zeta - zeta_mean)**2)/(k-1)
    C_mean = C_array.mean()
    hat_asvarB_sq = hat_asvarA_sq/(C_mean**2)

    z_val = norm.ppf(1 - 0.05/2)

    # CI for X(t_max)
    X_halfwidth = z_val*math.sqrt(hat_asvarA_sq)/math.sqrt(t_max)
    X_low = hatX_final - X_halfwidth
    X_high= hatX_final + X_halfwidth

    # CI for I_k
    I_halfwidth = z_val*math.sqrt(hat_asvarB_sq)/math.sqrt(k)
    I_low = hatI_final - I_halfwidth
    I_high= hatI_final + I_halfwidth

    return (X_low, X_high, I_low, I_high)


def main():
    args = ParseArguments()
    os.makedirs(args.results_path, exist_ok=True)

    # Setup RNG
    seed_str = args.seed.lower()
    if seed_str=="none":
        rng = np.random.default_rng()
        print("Using a random seed (none).")
    else:
        seed_val = int(seed_str)
        rng = np.random.default_rng(seed_val)
        print(f"Using seed={seed_val}")

    t_max = args.t_max
    t_zoom= args.t_zoom

    # ---------- Option 1: Exponential on/off ----------
    E_on1 = args.exp_on   # default 3.0
    E_off1= args.exp_off  # default 1.0
    p_true1= E_on1/(E_on1+ E_off1)

    print(f"\nOption1: Exponential ON(mean={E_on1}), OFF(mean={E_off1}). p_true={p_true1:.5f}")

    def on_1(): return rng.exponential(E_on1)
    def off_1():return rng.exponential(E_off1)

    (hX_times_1, hX_vals_1,
     hI_times_1, hI_vals_1,
     Z_1, C_1,
     cumul_1)= simulate_on_off(rng,t_max,on_1,off_1)

    hatX_fin_1 = cumul_1/t_max
    k1= len(Z_1)
    if k1>0:
        sumZ1= sum(Z_1)
        sumC1= sum(C_1)
        hatI_fin_1= sumZ1/sumC1
    else:
        hatI_fin_1=0

    ci1= compute_CI(t_max,Z_1,C_1,hatX_fin_1, hatI_fin_1)
    if ci1 is not None:
        (xlow1, xhigh1, Ilow1, Ihigh1)= ci1
    else:
        xlow1=xhigh1=Ilow1=Ihigh1=float('nan')

    # produce 2 plots
    def makeplot(x_t, x_v, i_t, i_v, ptrue,
                 outpdf, xlim, titlestr):
        plt.figure(figsize=(8,3))
        plt.plot(x_t, x_v, label=r"$\hat{X}(t)$")
        plt.step(i_t, i_v, where='post', label=r"$\hat{I}_k$")
        plt.axhline(y=ptrue, color='r', ls='--', label="True p")
        plt.xlim([0, xlim])
        plt.ylim([0.6,1.0])  # user wants y in [0.6,1.0]
        plt.grid(True)
        plt.legend()
        plt.savefig(outpdf, bbox_inches='tight')
        plt.title(titlestr)
        plt.xlabel("Time")
        plt.ylabel("Estimators")
        print(f"Saved {outpdf}")

    out1zoom = os.path.join(args.results_path, "ch8_on_off_option1_Exp_zoom.pdf")
    makeplot(hX_times_1, hX_vals_1, hI_times_1, hI_vals_1,
             p_true1, out1zoom, t_zoom, "Option1 Exp/Exp (zoom)")

    out1full = os.path.join(args.results_path, "ch8_on_off_option1_Exp_full.pdf")
    makeplot(hX_times_1, hX_vals_1, hI_times_1, hI_vals_1,
             p_true1, out1full, t_max, "Option1 Exp/Exp (full)")

    # ---------- Option 2: Pareto/Gamma ----------
    alpha_par = args.pareto_alpha   # default 2.5
    xm_par    = args.pareto_scale   # default 1.8
    # mean of Pareto = alpha*xm/(alpha-1), for alpha>1
    E_on2= alpha_par*xm_par/(alpha_par-1)
    k_g = args.gamma_shape   # default 2.0
    th_g= args.gamma_scale   # default 0.5
    E_off2= k_g*th_g
    p_true2= E_on2/(E_on2+E_off2)

    print(f"\nOption2: Pareto on(alpha={alpha_par},scale={xm_par}), Gamma off(k={k_g},th={th_g})")
    print(f" => E_on2={E_on2:.4f}, E_off2={E_off2:.4f}, p_true={p_true2:.5f}")

    def on_2(): return pareto.rvs(alpha_par, scale=xm_par, random_state=rng)
    def off_2():return gamma.rvs(k_g, scale=th_g, random_state=rng)

    (hX_times_2, hX_vals_2,
     hI_times_2, hI_vals_2,
     Z_2, C_2, cumul_2) = simulate_on_off(rng,t_max, on_2, off_2)

    hatX_fin_2= cumul_2/t_max
    k2= len(Z_2)
    if k2>0:
        sumZ2= sum(Z_2)
        sumC2= sum(C_2)
        hatI_fin_2= sumZ2/sumC2
    else:
        hatI_fin_2=0

    ci2= compute_CI(t_max, Z_2, C_2, hatX_fin_2, hatI_fin_2)
    if ci2 is not None:
        (xlow2, xhigh2, Ilow2, Ihigh2)= ci2
    else:
        xlow2=xhigh2=Ilow2=Ihigh2=float('nan')

    out2zoom= os.path.join(args.results_path,"ch8_on_off_option2_Par_zoom.pdf")
    makeplot(hX_times_2, hX_vals_2, hI_times_2, hI_vals_2,
             p_true2, out2zoom, t_zoom, "Option2: Pareto/Gamma (zoom)")

    out2full= os.path.join(args.results_path,"ch8_on_off_option2_Par_full.pdf")
    makeplot(hX_times_2, hX_vals_2, hI_times_2, hI_vals_2,
             p_true2, out2full, t_max, "Option2: Pareto/Gamma (full)")

    # Compare means across options:
    # "By 'being equal' I mean if mean(off, exp)=mean(off, pareto) & mean(on, exp)=mean(on, pareto)"
    same_on  = np.isclose(E_on1, E_on2, rtol=1e-7)
    same_off = np.isclose(E_off1, E_off2, rtol=1e-7)
    msg_on   = "SAME" if same_on else "NOT the same"
    msg_off  = "SAME" if same_off else "NOT the same"

    print(f"\nCompare Option1 vs Option2 means:\n  On-time means: {E_on1} vs {E_on2:.4f} => {msg_on}")
    print(f"  Off-time means: {E_off1} vs {E_off2:.4f} => {msg_off}\n")

    def ci_str(lo, hi):
        if math.isnan(lo) or math.isnan(hi):
            return "N/A"
        return f"({lo:.3f}, {hi:.3f})"

    # Build final table
    data_table = []
    data_table.append(["True value of p", f"{p_true1:.5f}", f"{p_true2:.5f}"])
    data_table.append(["Number of cycles(k)", f"{k1:d}", f"{k2:d}"])
    data_table.append(["Estimator X(t_max)", f"{hatX_fin_1:.5f}", f"{hatX_fin_2:.5f}"])
    data_table.append(["Estimator I_k", f"{hatI_fin_1:.5f}", f"{hatI_fin_2:.5f}"])
    data_table.append(["95% CI for X(t)", ci_str(xlow1,xhigh1), ci_str(xlow2,xhigh2)])
    data_table.append(["95% CI for I_k", ci_str(Ilow1,Ihigh1), ci_str(Ilow2,Ihigh2)])

    table_str = tabulate(data_table, headers=["", "Option1(Exp/Exp)", "Option2(Par/Gam)"], tablefmt="github")
    print("\nFinal Summary Table:\n")
    print(table_str)

    plt.show()


if __name__=="__main__":
    main()
