import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Read the n values from the file with error handling
try:
    with open('so_far.txt', 'r') as file:
        data = file.readlines()
    n_values = [int(line.strip()) for line in data if line.strip().isdigit()]
    if not n_values:
        raise ValueError("No valid integer data found.")
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

# Ensure all n_values are positive for exponential fitting
if any(n <= 0 for n in n_values):
    print("All n values must be positive for exponential fitting.")
    exit(1)

# Convert n_values to numpy array
indices = np.arange(1, len(n_values) + 1)
n_values_array = np.array(n_values)

# Define exponential function
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Improved initial parameter guesses
a_initial = n_values_array[0]
b_initial = np.log(n_values_array[-1]/a_initial) / (indices[-1] - indices[0])
initial_guess = [a_initial, b_initial]

# Fit the exponential model with bounds to ensure realistic parameters
try:
    params, covariance = curve_fit(
        exp_func, 
        indices, 
        n_values_array, 
        p0=initial_guess, 
        bounds=(0, np.inf),
        maxfev=10000
    )
except RuntimeError as e:
    print(f"Curve fitting failed: {e}")
    exit(1)
except Exception as e:
    print(f"An error occurred during curve fitting: {e}")
    exit(1)

# Calculate R-squared and Mean Squared Error
residuals = n_values_array - exp_func(indices, *params)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((n_values_array - np.mean(n_values_array))**2)
r_squared = 1 - (ss_res / ss_tot)
mse = ss_res / len(n_values_array)
print(f"R-squared: {r_squared:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Deliminater
print("=+" * 20, "=", sep="")

# Predict n values for new indices
new_indices = np.arange(len(n_values) + 1, len(n_values) + 21)
predicted_n_values = exp_func(new_indices, *params)

# Print the estimated n values
print('Estimated n values:')
for i, n in enumerate(predicted_n_values, start=1):
    print(f'{len(n_values) + i}: {n:.0f}')

# Plot the actual and estimated n values with confidence intervals
plt.figure(figsize=(12, 7))
plt.scatter(indices, n_values_array, color='blue', label='Actual n values')
plt.plot(indices, exp_func(indices, *params), 'b-', label='Exponential Fit')
plt.scatter(new_indices, predicted_n_values, color='red', label='Estimated n values')
plt.plot(new_indices, predicted_n_values, 'r--')

# Calculate confidence intervals
alpha = 0.05
n = len(indices)
p = len(params)
dof = max(0, n - p)
t = 2.045  # t-score for 95% confidence with dof ~ 30
sigma = np.sqrt(np.diag(covariance))
conf_upper = exp_func(indices, *params) + t * sigma[0] * np.exp(params[1] * indices)
conf_lower = exp_func(indices, *params) - t * sigma[0] * np.exp(params[1] * indices)
plt.fill_between(indices, conf_lower, conf_upper, color='blue', alpha=0.2, label='95% Confidence Interval')

plt.xlabel('Index')
plt.ylabel('n')
plt.title('Actual and Estimated n values with Exponential Fit')
plt.legend()
plt.grid(True)
plt.show()
