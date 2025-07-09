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
## File: ch5_cross_entropy_normal_1D_optimal_mu_s2.py               ##
##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##==##




import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize

from tabulate import tabulate


# Target function for probability estimation
def target_prob(x):
    if np.isscalar(x):
        return 1.0 if x > 4 else 0.0
    else:
        return (x > 4).astype(float)

# Standard normal density
def f_standard(x):
    return norm.pdf(x)

# General normal density
def f_general(x, mu, s2):
    sigma = np.sqrt(s2)
    return norm.pdf(x, loc=mu, scale=sigma)

# Define cross-entropy objective: maximize expected log f_general under standard normal conditioned on X>4
def cross_entropy_objective(mu_s2, fix_s2=False):
    mu = mu_s2[0]
    if fix_s2:
        s2 = 1.0
    else:
        s2 = mu_s2[1]
    if s2 <= 0:
        return np.inf  # Penalize invalid s2

    def integrand(x):
        log_f = -0.5*np.log(2*np.pi*s2) - ((x-mu)**2)/(2*s2)
        return log_f * f_standard(x)

    integral, _ = quad(integrand, 4, 20, epsabs=1e-10, epsrel=1e-10)
    return -integral  # because minimize

# Find optimal parameters
def find_optimal_parameters(fix_s2=False, s2_lower_bound=None):
    x0 = [5.0, 2.0] if not fix_s2 else [5.0]
    constraints = []
    if not fix_s2 and s2_lower_bound is not None:
        constraints.append({'type': 'ineq', 'fun': lambda x: x[1] - s2_lower_bound})

    result = minimize(
        lambda params: cross_entropy_objective(params, fix_s2=fix_s2),
        x0,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-12, 'disp': False, 'maxiter': 1000}
    )

    if result.success:
        if fix_s2:
            return result.x[0], 1.0
        else:
            return result.x[0], result.x[1]
    else:
        raise RuntimeError("Optimization failed: " + result.message)

# Theoretical second moment of IS estimator
def theoretical_second_moment(mu, s2):
    def integrand(x):
        numerator = f_standard(x) * target_prob(x)
        denominator = f_general(x, mu, s2)
        return (numerator / denominator)**2 * f_general(x, mu, s2)

    value, _ = quad(integrand, -10, 20, epsabs=1e-10, epsrel=1e-10)
    return value

# Perform simulations
def simulate_estimator(mu, s2, R=10000):
    rng = np.random.default_rng(12345)
    sigma = np.sqrt(s2)
    samples = rng.normal(loc=mu, scale=sigma, size=R)
    weights = f_standard(samples) / f_general(samples, mu, s2)
    estimates = weights * target_prob(samples)
    estimate_mean = np.mean(estimates)
    estimate_var = np.var(estimates, ddof=1)
    estimate_second_moment = np.mean(estimates**2)
    return estimate_mean, estimate_var, estimate_second_moment

# MAIN PART
if __name__ == "__main__":
    R = 10000

    # Step 1: find parameters
    mu1, s2_fixed1 = find_optimal_parameters(fix_s2=True)  # fixed s2=1
    mu2, s2_norestr = find_optimal_parameters(fix_s2=False)  # no restriction
    mu3, s2_restr = find_optimal_parameters(fix_s2=False, s2_lower_bound=0.5)  # s2>=0.5

    print(f"Optimal mu (fixed s2=1): {mu1:.6f}")
    print(f"Optimal mu (no restriction): {mu2:.6f}, s2: {s2_norestr:.6f}")
    print(f"Optimal mu (with s2>=0.5): {mu3:.6f}, s2: {s2_restr:.6f}")
    print("-" * 50)

    # Step 2: theoretical second moments
    moments = {}

    # a) CMC (Crude Monte Carlo)
    true_p, _ = quad(lambda x: f_standard(x), 4, np.inf)
    var_cmc = true_p * (1 - true_p)
    moments["CMC"] = (true_p, var_cmc, true_p)  # Second moment is true_p for indicator


    # b) IS with (mu1, 1)
    second_moment_b = theoretical_second_moment(mu1, 1)
    var_b = second_moment_b - true_p**2
    moments["IS (mu1, 1)"] = (true_p, var_b, second_moment_b)


    # c) IS with (mu2, s2_norestr)
    second_moment_c = theoretical_second_moment(mu2, s2_norestr)
    var_c = second_moment_c - true_p**2
    moments["IS (mu2, s2_norestr)"] = (true_p, var_c, second_moment_c)


    # d) IS with (mu3, s2=0.5)
    second_moment_d = theoretical_second_moment(mu3, s2_restr)
    var_d = second_moment_d - true_p**2
    moments["IS (mu3, s2_restr=0.5)"] = (true_p, var_d, second_moment_d)

    # Step 3: Simulations
    simulations = {}

    rng = np.random.default_rng(12345)

    # a) CMC
    samples_cmc = rng.standard_normal(R)
    estimates_cmc = (samples_cmc > 4).astype(float)
    simulations["CMC"] = (np.mean(estimates_cmc), np.var(estimates_cmc, ddof=1), np.mean(estimates_cmc**2))


    # b) IS (mu1, 1)
    simulations["IS (mu1, 1)"] = simulate_estimator(mu1, 1, R)

    # c) IS (mu2, s2_norestr)
    simulations["IS (mu2, s2_norestr)"] = simulate_estimator(mu2, s2_norestr, R)

    # d) IS (mu3, s2=0.5)
    simulations["IS (mu3, s2_restr=0.5)"] = simulate_estimator(mu3, s2_restr, R)

    # Step 4: Display results
    # Build table rows
    table = []
    for method in moments:
        true_mean, true_var, true_second_moment = moments[method]
        est_mean, est_var, est_second_moment = simulations[method]
        table.append([
            method,
            f"{true_mean:.6e}",
            f"{true_var:.6e}" if not np.isnan(true_var) else "nan",
            f"{true_second_moment:.6e}" if not np.isnan(true_second_moment) else "nan",
            f"{est_mean:.6e}",
            f"{est_var:.6e}",
            f"{est_second_moment:.6e}",
        ])

    headers = ["Method", "True Mean", "True Variance", "True Second Moment", "Estimated Mean", "Estimated Variance", "Estimated Second Moment"]
    print(tabulate(table, headers=headers, tablefmt="grid"))


