import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Read the n values from the file
with open('so_far.txt', 'r') as file:
    data = file.readlines()

# Convert the data to a list of integers
n_values = [int(line.strip()) for line in data]

# Convert n_values to a numpy array and reshape for polynomial regression
indices = np.array(range(len(n_values))).reshape(-1, 1)
n_values_array = np.array(n_values)

# Find the best polynomial degree for the regression
best_degree = 0
best_score = 0
for degree in range(1, 11):
    poly = PolynomialFeatures(degree=degree)
    indices_poly = poly.fit_transform(indices)
    model = LinearRegression()
    model.fit(indices_poly, n_values_array)
    score = model.score(indices_poly, n_values_array)
    if score > best_score:
        best_score = score
        best_degree = degree

print(f'Best polynomial degree: {best_degree} with score: {best_score:.2f}')

# Perform polynomial regression
poly = PolynomialFeatures(degree=best_degree)  # You can adjust the degree as needed
indices_poly = poly.fit_transform(indices)
model = LinearRegression()
model.fit(indices_poly, n_values_array)

# Predict n values for new indices
new_indices = np.arange(len(n_values), len(n_values) + 20).reshape(-1, 1)
new_indices_poly = poly.transform(new_indices)
predicted_n_values = model.predict(new_indices_poly)

# Plot the actual and estimated n values
plt.figure(figsize=(10, 6))
plt.plot(range(len(n_values)), n_values, 'bo', label='Actual Lewis Primes n values')
plt.plot(range(len(n_values)), model.predict(indices_poly), 'b-', label='Polynomial Fit')
plt.plot(range(len(n_values), len(n_values) + 20), predicted_n_values, 'ro', label='Estimated n values')
plt.plot(range(len(n_values), len(n_values) + 20), model.predict(new_indices_poly), 'r-', label='Estimated Polynomial Fit')
plt.xlabel('Index')
plt.ylabel('n')
plt.title('Lewis Primes n values and Estimated n values')
plt.legend()
plt.grid(True)
plt.show()

# Print the estimated n values
print('Estimated n values:')
for i, n in enumerate(predicted_n_values):
    print(f'{len(n_values) + i}: {n:.0f}')