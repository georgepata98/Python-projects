import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing
from sklearn.metrics import mean_squared_error, r2_score

i = 0

def entropy(f, f_def):
    """ Shannon entropy S as defined in Reginatto (2002). """
    return -np.sum(f * np.log(f / f_def) + f_def - f)

def compute_spectrum(R, f_def, lambdas):
    """ Compute spectrum using lambda multipliers. """
    exponent = -np.dot(lambdas, R)
    return f_def * np.exp(exponent)

def chi_squared(f, R, N, epsilon):
    """ Chi-squared with uncertainties epsilon_k. """
    predicted_N = np.dot(R, f)
    return np.sum(((N - predicted_N) / epsilon) ** 2)

def constraint_function(lambdas, R, f_def, N, epsilon, O):
    f = compute_spectrum(R, f_def, lambdas)
    chi2 = chi_squared(f, R, N, epsilon)
    return (chi2 - O) / O  # Normalized deviation from expected chi2

def objective(lambdas, R, f_def, N, epsilon, O, penalty_factor):
    global i
    f = compute_spectrum(R, f_def, lambdas)
    S = entropy(f, f_def)
    constraint_val = constraint_function(lambdas, R, f_def, N, epsilon, O)
    penalty = penalty_factor * (constraint_val) ** 2
    i = i + 1
    return -S + penalty


# Synthetic test data for 4 Bonner spheres and 8 energy bins
energy_bins = 8
f_true = np.array([10, 30, 50, 80, 60, 40, 20, 10])

R = np.array([
    [0.6, 0.9, 1.0, 0.5, 0.3, 0.1, 0.05, 0.02],
    [0.2, 0.4, 0.9, 1.0, 0.8, 0.5, 0.2, 0.1],
    [0.05, 0.1, 0.3, 0.8, 1.0, 0.9, 0.6, 0.3],
    [0.01, 0.03, 0.1, 0.3, 0.8, 1.0, 0.9, 0.6]
])

N = R @ f_true
epsilon = 0.1 * N
f_def = np.ones(energy_bins)

# Expected value of chi-squared
O = np.sum((N / epsilon) ** 2)

# Initial guess for Lagrange multipliers λ_k
lambdas_initial = np.zeros(len(N))
penalty_factor = 1e3 * O  # scale for penalty term

# Conjugate Gradient Optimization
result_cg = minimize(
    objective, lambdas_initial,
    args=(R, f_def, N, epsilon, O, penalty_factor),
    method='CG'
)

# Simulated Annealing Optimization
bounds = [(-5, 5)] * len(N)
result_sa = dual_annealing(
    objective,
    bounds=bounds,
    args=(R, f_def, N, epsilon, O, penalty_factor)
)

# Extract results
optimal_lambdas_cg = result_cg.x
f_unfolded_cg = compute_spectrum(R, f_def, optimal_lambdas_cg)

optimal_lambdas_sa = result_sa.x
f_unfolded_sa = compute_spectrum(R, f_def, optimal_lambdas_sa)

entropy_cg = entropy(f_unfolded_cg, f_def)
entropy_sa = entropy(f_unfolded_sa, f_def)

predicted_N_cg = np.dot(R, f_unfolded_cg)
predicted_N_sa = np.dot(R, f_unfolded_sa)

# Printare rezultate
print("\n--- Results using Conjugate Gradient (CG) ---")
print("Optimal lambdas (CG):", optimal_lambdas_cg)
print("Unfolded spectrum, f_i (CG):", f_unfolded_cg)

print("\n--- Results using Simulated Annealing (SA) ---")
print("Optimal lambdas (SA):", optimal_lambdas_sa)
print("Unfolded spectrum, f_i (SA):", f_unfolded_sa)

mse_cg = mean_squared_error(N, predicted_N_cg)
r2_cg = r2_score(N, predicted_N_cg)

mse_sa = mean_squared_error(N, predicted_N_sa)
r2_sa = r2_score(N, predicted_N_sa)

# Plotting
plot = False
if plot == True:
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.bar(range(len(f_unfolded_cg)), f_unfolded_cg, color='blue', alpha=0.7)
    plt.title(f'Unfolded Spectrum - CG\nEntropy: {entropy_cg:.4f}, MSE: {mse_cg:.4f}, R²: {r2_cg:.4f}')
    plt.xlabel('Energy Bin')
    plt.ylabel('Spectrum Value')

    plt.subplot(1, 3, 2)
    plt.bar(range(len(f_unfolded_sa)), f_unfolded_sa, color='green', alpha=0.7)
    plt.title(f'Unfolded Spectrum - SA\nEntropy: {entropy_sa:.4f}, MSE: {mse_sa:.4f}, R²: {r2_sa:.4f}')
    plt.xlabel('Energy Bin')
    plt.ylabel('Spectrum Value')

    residuals_cg = (N - predicted_N_cg) / epsilon
    residuals_sa = (N - predicted_N_sa) / epsilon

    plt.subplot(1, 3, 3)
    plt.plot(residuals_cg, label='CG Residuals', marker='o')
    plt.plot(residuals_sa, label='SA Residuals', marker='x')
    plt.title('Residuals Comparison')
    plt.xlabel('Detector Index')
    plt.ylabel('Normalized Residuals')
    plt.legend()

    plt.tight_layout()
    plt.show()
