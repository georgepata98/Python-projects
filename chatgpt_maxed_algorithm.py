# Exemplu de la ChatGPT cu metoda MaxEnt din Reginatto et al. 2002
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, dual_annealing

def entropy(f, f_def):
    """ Calculate entropy function S. """
    return -np.sum(f*np.log(f/f_def)+f_def-f)

def compute_spectrum(R, f_def, lambdas):
    """ Compute the spectrum f_i using the given lambdas. """
    exponent = -np.dot(lambdas, R)  # np.dot face inner product pt. 1D arrays si matrix multiplication pt. 2D arrays
    return f_def*np.exp(exponent)

def constraint_function(lambdas, R, N, s, O):
    """ Compute the constraint function based on measured data N. """
    f = compute_spectrum(R, f_def, lambdas)
    predicted_N = np.dot(R, f)
    residuals = (N-predicted_N)/s
    chi_squared = np.sum(residuals**2)
    return (chi_squared-O)/O

def objective(lambdas, R, f_def, N, s, O, penalty_factor):
    """ Objective function to be minimized using lambda. """
    f = compute_spectrum(R, f_def, lambdas)
    S = entropy(f, f_def)
    constraint_value = constraint_function(lambdas, R, N, s, O)
    penalty = penalty_factor*(constraint_value)**2  # large penalty if constraint is not satisfied
    return -S+penalty



# EXEMPLU input data:
R = np.array([[0.2, 0.1], [0.4, 0.3], [0.4, 0.6]])  # matricea de raspuns m x n
f_def = np.array([1.0, 1.0])   # spectrul default de dimensiune n
N = np.array([0.5, 1.0, 1.5])  # datele masurate de dimensiune m
s = np.array([0.1, 0.1, 0.1])  # incertitudinile standard ale datelor masurate, dimensiune m
O = np.sum((N/s)**2)  # de obicei omega = nr. detectori

# Ghicirea initiala a lambdelor
lambdas_initial = np.zeros(len(N))  # se creaza un array 1D de zerouri cu lungime egala cu nr. de detectori

# Penalty factor dinamic
penalty_factor = 1e3*O

# Se face optimizarea folosind metoda Conjugate Gradient (CG)
result_cg = minimize(objective, lambdas_initial, args=(R, f_def, N, s, O, penalty_factor), method='CG')

# Se face optimizarea folosind methoda Simulated Annealing (SA)
bounds = [(-5, 5)]*len(N)  # example bonds for each lambda, adjust as needed
result_sa = dual_annealing(objective, bounds=bounds, args=(R, f_def, N, s, O, penalty_factor))

# Extragere lambde optime si calculare unfolded spectrum pentru ambele metode
optimal_lambdas_cg = result_cg.x
f_unfolded_cg = compute_spectrum(R, f_def, optimal_lambdas_cg)
optimal_lambdas_sa = result_sa.x
f_unfolded_sa = compute_spectrum(R, f_def, optimal_lambdas_sa)

# Printare rezultate
print("\n--- Results using Conjugate Gradient (CG) ---")
print("Optimal lambdas (CG):", optimal_lambdas_cg)
print("Unfolded spectrum (CG):", f_unfolded_cg)

print("\n--- Results using Simulated Annealing (SA) ---")
print("Optimal lambdas (SA):", optimal_lambdas_sa)
print("Unfolded spectrum (SA):", f_unfolded_sa)

# Calcul entropii pentru comparare
entropy_cg = entropy(f_unfolded_cg, f_def)
entropy_sa = entropy(f_unfolded_sa, f_def)



# Plotare rezultate pentru comparare
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.bar(range(len(f_unfolded_cg)), f_unfolded_cg, color='blue', alpha=0.7)
plt.title(f'Unfolded Spectrum - CG\nEntropy: {entropy_cg:.4f}')
plt.xlabel('Index')
plt.ylabel('Spectrum value')

plt.subplot(1, 3, 2)
plt.bar(range(len(f_unfolded_sa)), f_unfolded_sa, color='green', alpha=0.7)
plt.title(f'Unfolded Spectrum - SA\nEntropy: {entropy_sa:.4f}')
plt.xlabel('Index')
plt.ylabel('Spectrum value')

# Vizualizare reziduali
predicted_N_cg = np.dot(R, f_unfolded_cg)
predicted_N_sa = np.dot(R, f_unfolded_sa)
residuals_cg = (N-predicted_N_cg)/s
residuals_sa = (N-predicted_N_sa)/s

plt.subplot(1, 3, 3)
plt.plot(residuals_cg, label='CG Residuals', marker='o')
plt.plot(residuals_sa, label='SA Residuals', marker='x')
plt.title('Residuals Comparison')
plt.xlabel('Data Index')
plt.ylabel('Normalized Residuals')
plt.legend()

plt.tight_layout()
plt.show()