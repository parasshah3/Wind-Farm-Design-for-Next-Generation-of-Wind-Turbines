import numpy as np
import matplotlib.pyplot as plt
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

# Generate a smooth range of heights and predicted wind speeds
smooth_heights = np.linspace(10, 550, 500)  # Smooth range of heights (from 10m to 550m)
predicted_wind_speeds = slope * np.log(smooth_heights / z0)  # U(z) = A * ln(z / z0)

# Plot original data and fitted wind speed profile
plt.figure(figsize=(10, 6))

# Scatter plot of measured wind speeds
plt.scatter(heights, wind_speeds, color='blue', label='Measured Wind Speeds', zorder=3)

# Smooth line for fitted relationship
plt.plot(smooth_heights, predicted_wind_speeds, color='red', linestyle='--', linewidth=1,
         label=f'Fitted Relationship: A={slope:.3f}, z0={z0:.3e} m', zorder=2)

# Plot settings
plt.xlabel('Height (m)')
plt.ylabel('Mean Wind Speed (m/s)')
plt.title('Comparison of Measured Wind Speeds and Fitted Logarithmic Relationship')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Print calculated values
print("Calculated Parameters:")
print(f"Slope (A): {slope:.6f} m/s")
print(f"Roughness Length (z0): {z0:.6e} m")
print(f"R-squared: {r_value**2:.4f}")