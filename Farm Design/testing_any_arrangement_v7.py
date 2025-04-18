import farm_design_functions_v6 as fd  # Import the module with your functions
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the test (all in metres)
v_i = 9.519  # Freestream wind speed (m/s)
Ct = 0.761  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
D = 2 * r_0  # Rotor diameter (m)
z = 100  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# Define turbine positions based on the given arrangement
A = 2 * D  # Scaling factor for turbine spacing
turbine_positions = {
    1: (0 * A, 0 * A),    # Turbine 1
    2: (1 * A, 1 * A),    # Turbine 2
    3: (1 * A, -1 * A),   # Turbine 3
    4: (2 * A, 0 * A),    # Turbine 4
    5: (3 * A, 1 * A),    # Turbine 5
    6: (3 * A, -1 * A),   # Turbine 6
    7: (4 * A, 0 * A),    # Turbine 7
}

# Calculate effective wind speeds using `mwm_identical_turbines`
effective_wind_speeds = fd.mwm_identical_turbines(v_i, Ct, r_0, z, z_0, turbine_positions)

# Plot the turbine arrangement
x_coords = [pos[0] / D for pos in turbine_positions.values()]  # Convert to rotor diameters for plotting
y_coords = [pos[1] / D for pos in turbine_positions.values()]

# Determine dynamic plot limits
x_min, x_max = min(x_coords), max(x_coords)
y_min, y_max = min(y_coords), max(y_coords)

# Add padding to plot limits
padding = 0.5  # Padding in rotor diameters
x_lim = (x_min - padding, x_max + padding)
y_lim = (y_min - padding, y_max + padding)

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(x_coords, y_coords, color="black", s=100, label="Turbines")

# Add labels for each turbine including calculated effective wind speed
for turbine_id, (x, y) in turbine_positions.items():
    v_j = effective_wind_speeds[turbine_id]
    ax.text(x / D, y / D + 0.2, f"{turbine_id} (v_j={v_j:.4f} m/s)", fontsize=10, ha="center", color="blue")

# Add incoming wind arrows (v0)
for y_arrow in np.arange(y_lim[0], y_lim[1], 0.5):  # Adjust range dynamically
    ax.arrow(x_lim[0] - 0.2, y_arrow, 0.4, 0, head_width=0.1, head_length=0.1, fc="black", ec="black")

# Add v0 label near the arrows
ax.text(x_lim[0] - 0.3, y_lim[1] + 0.2, f"$v_0 = {v_i:.4f}$ m/s", fontsize=12, color="black")

# Formatting
ax.set_title("Wind Turbine Arrangement with Effective Wind Speeds", fontsize=14)
ax.set_xlabel("Downstream Distance (Rotor Diameters)", fontsize=12)
ax.set_ylabel("Perpendicular Distance (Rotor Diameters)", fontsize=12)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)  # Horizontal axis
ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)  # Vertical axis
ax.grid(color="gray", linestyle="--", linewidth=0.5)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax.set_aspect("equal", adjustable="box")
plt.legend(loc="upper left", fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()

# Print results for each turbine
print("\nResults from `mwm_identical_turbines` for the given arrangement:")
for turbine_id, wind_speed in effective_wind_speeds.items():
    print(f"Turbine {turbine_id}: {wind_speed:.4f} m/s")