from farm_design_functions_v12 import mwm_identical_turbines_BEM_any_direction
import matplotlib.pyplot as plt

# File path and sheet name
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# Define turbine arrangement and parameters
Ct = 0.806  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
z = 140  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)
A = 1.5 * (2 * r_0)  # Spacing in multiples of rotor diameter
turbine_positions = {
    1: (0 * A, 0 * A),
    2: (1 * A, 0.5 * A),
    3: (1 * A, -0.5 * A),
    4: (2 * A, 0 * A),
    5: (3 * A, 0.5 * A),
    6: (3 * A, -0.5 * A),
    7: (4 * A, 0 * A),
}

# Define range of theta values (in degrees)
theta_values = [5, 10, 15, 20]  # Angles to test

# Create subplots for all angles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, theta in enumerate(theta_values):
    print(f"\nCalculating for theta = {theta}°...")
    results = mwm_identical_turbines_BEM_any_direction(file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, theta)

    # Extract data for plotting
    turbine_ids = list(results.keys())
    avg_yield = [metrics["Average Energy Yield (MW/hr)"] for metrics in results.values()]

    # Plot on the corresponding subplot
    axes[i].bar(turbine_ids, avg_yield, color="skyblue")
    axes[i].set_title(f"θ = {theta}°")
    axes[i].set_xlabel("Turbine ID")
    axes[i].set_ylabel("Energy Yield (MW/hr)")
    axes[i].set_ylim(0, max(avg_yield) + 2)  # Add some padding to the y-axis
    axes[i].grid(axis="y", linestyle="--", linewidth=0.5)
    axes[i].set_xticks(turbine_ids)  # Ensure turbine IDs are shown on the x-axis

# Adjust layout for better visibility
plt.suptitle("Average Energy Yield for Various Wind Directions (θ)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add space for the main title
plt.show()