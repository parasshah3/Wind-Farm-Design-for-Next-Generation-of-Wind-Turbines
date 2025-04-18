import numpy as np
import matplotlib.pyplot as plt
from farm_design_functions_v21 import genetic_algorithm

# Define file path to BEM spreadsheet
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# Define wind farm boundary (UTM coordinates)
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Closing the boundary loop
])

# Define wind turbine parameters
Ct = 0.806  # Thrust coefficient
r_0 = 124.7  # Rotor radius (m)
z = 140  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)
min_spacing = 2 * 2 * r_0  # Minimum spacing (3D rule)

# Wind rose data
wind_rose = {
    0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
    180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
}
primary_direction = 240  # Most common wind direction

# Define constraints for turbine numbers
num_turbines_range = (35, 85)  # Min & max turbines

# Run the Genetic Algorithm
best_layouts, final_best_layout, final_best_LCOE = genetic_algorithm(
    file_path, sheet_name, boundary_utm, min_spacing, num_turbines_range,
    Ct, r_0, z, z_0, wind_rose, max_generations=50, convergence_threshold=0.001
)

# Print top 5 layouts
print("\nTop 5 Optimal Wind Farm Layouts:")
for i, (layout, lcoe, gen) in enumerate(best_layouts, 1):
    print(f"Rank {i}: LCOE = {lcoe:.2f} £/MWhr (Generation {gen}) | {len(layout)} turbines")

# Print final best layout details
print("\nFinal Best Wind Farm Layout:")
print(f"LCOE: {final_best_LCOE:.2f} £/MWhr")
print(f"Number of turbines: {len(final_best_layout)}")

# Plot LCOE Evolution Over Generations
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(best_layouts) + 1), [lcoe for _, lcoe, _ in best_layouts], marker='o', linestyle='-')
plt.xlabel("Generation")
plt.ylabel("LCOE (£/MWhr)")
plt.title("LCOE Evolution Over Generations")
plt.grid()
plt.show()

# Visualize the final best wind farm layout
plt.figure(figsize=(8, 8))
plt.plot(boundary_utm[:, 0], boundary_utm[:, 1], 'b-', label="Wind Farm Boundary")

# Extract turbine positions
turbine_x, turbine_y = zip(*final_best_layout.values())

# Plot turbines
plt.scatter(turbine_x, turbine_y, c='r', marker='o', label=f"Turbines ({len(final_best_layout)})")

# Annotate turbine numbers
for turbine_id, (x, y) in final_best_layout.items():
    plt.text(x, y + 20, str(turbine_id), fontsize=8, ha="center", color="black")

plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title("Final Best Wind Farm Layout")
plt.legend()
plt.grid()
plt.show()