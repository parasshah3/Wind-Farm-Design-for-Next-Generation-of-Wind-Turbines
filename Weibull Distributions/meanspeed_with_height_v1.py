import numpy as np
from scipy.stats import linregress

# Input data: heights (z) in meters and mean wind speed (U) in m/s
heights = np.array([50, 75, 100, 150, 200, 250, 500])  # Heights (z)
wind_speeds = np.array([8.556808472, 8.854023933, 9.075548172, 
                        9.396860123, 9.631995201, 9.813015938, 10.31694698])  # Wind speeds (U)

# Transform heights to natural logarithm
log_heights = np.log(heights)

# Perform linear regression to solve for A (slope) and intercept
slope, intercept, r_value, p_value, std_err = linregress(log_heights, wind_speeds)

# Calculate roughness length (z0) from intercept
z0 = np.exp(-intercept / slope)

# Print the results
print("Linear Regression Results:")
print(f"A (slope): {slope:.6f} m/s")
print(f"Intercept (A * ln(z0)): {intercept:.6f}")
print(f"Roughness Length (z0): {z0:.10e} m")
print(f"R-squared: {r_value**2:.4f}")

# Optional: Plot the data and the fitted line
import matplotlib.pyplot as plt

# Predicted wind speeds
U_pred = slope * log_heights + intercept

plt.figure(figsize=(8, 6))
plt.scatter(log_heights, wind_speeds, color='blue', label='Measured Data')
plt.plot(log_heights, U_pred, color='red', label='Fitted Line (U = A ln(z/z0))')
plt.xlabel('ln(Height) [ln(z)]')
plt.ylabel('Mean Wind Speed [U(z)] (m/s)')
plt.title('Log-Linear Relationship Between Wind Speed and Height')
plt.legend()
plt.grid()
plt.show()