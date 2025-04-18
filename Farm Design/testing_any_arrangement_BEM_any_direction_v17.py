import matplotlib.pyplot as plt
from farm_design_functions_v12 import transform_turbine_positions

# Define turbine parameters and positions
rotor_diameter = 126 * 2  # Rotor diameter (m)
turbine_positions = {
    1: (0, 0),
    2: (4 * rotor_diameter, 1.5 * rotor_diameter),
    3: (4 * rotor_diameter, -1.5 * rotor_diameter),
    4: (8 * rotor_diameter, 0),
    5: (12 * rotor_diameter, 1.5 * rotor_diameter),
    6: (12 * rotor_diameter, -1.5 * rotor_diameter),
    7: (16 * rotor_diameter, 0)
}

# Wind angle
theta = 15 # Wind angle in degrees

# Transform turbine positions
transformed_positions = transform_turbine_positions(turbine_positions, theta)

# Convert positions to rotor diameters for plotting
x_original = [pos[0] / rotor_diameter for pos in turbine_positions.values()]
y_original = [pos[1] / rotor_diameter for pos in turbine_positions.values()]

x_transformed = [pos[0] / rotor_diameter for pos in transformed_positions.values()]
y_transformed = [pos[1] / rotor_diameter for pos in transformed_positions.values()]

# Plot original and transformed turbine positions
plt.figure(figsize=(10, 6))

# Plot original turbine positions
plt.scatter(x_original, y_original, color='blue', label='Original Positions', s=100)
for turbine_id, (x, y) in turbine_positions.items():
    plt.text(x / rotor_diameter, y / rotor_diameter + 0.2, f"{turbine_id}", color='blue', ha='center')

# Plot transformed turbine positions
plt.scatter(x_transformed, y_transformed, color='red', label='Transformed Positions', s=100)
for turbine_id, (x, y) in transformed_positions.items():
    plt.text(x / rotor_diameter, y / rotor_diameter + 0.2, f"{turbine_id}", color='red', ha='center')

# Add legend, grid, and labels
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend(loc='upper left')
plt.title(f"Original and Transformed Turbine Positions (θ = {theta}°)", fontsize=14)
plt.xlabel("x (Rotor Diameters)")
plt.ylabel("y (Rotor Diameters)")
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()