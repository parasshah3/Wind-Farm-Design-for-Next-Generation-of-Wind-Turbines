import numpy as np
import matplotlib.pyplot as plt
from farm_design_functions_v16 import initialise_population, next_generation

# **Define file path to BEM spreadsheet**
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# **Define wind turbine parameters**
Ct = 0.806  # Thrust coefficient
r_0 = 124.7  # Updated rotor radius (m)
z = 150  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# **Define wind rose and primary direction**
wind_rose = {
    0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
    180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
}
primary_direction = 240  # Most common wind direction

# **Define wind farm boundary (UTM coordinates)**
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Closing the boundary loop
])

# **Define constraints**
min_spacing = 2 * 2 * r_0  # Minimum spacing (3D)
population_size = 10  # Number of layouts per generation
num_turbines_range = (40, 60)  # Allowed range of turbine counts

# **Step 1: Initialize the first generation**
print("Generating Initial Population...\n")
initial_population = initialise_population(population_size, boundary_utm, min_spacing, num_turbines_range)

# **Step 2: Generate the next generation**
print("Generating Next Generation...\n")
new_population = next_generation(initial_population, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose, boundary_utm, min_spacing)

# **Step 3: Visualize 4 random layouts from the new generation**
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i in range(4):
    layout = new_population[i]
    turbine_x, turbine_y = zip(*layout.values())

    axes[i].plot(boundary_utm[:, 0], boundary_utm[:, 1], 'b-', label="Wind Farm Boundary")
    axes[i].scatter(turbine_x, turbine_y, c='r', marker='o', label=f"Turbines ({len(layout)})")

    for tid, (x, y) in layout.items():
        axes[i].text(x, y + 20, str(tid), fontsize=8, ha="center", color="black")

    axes[i].set_title(f"New Generation Layout {i+1}")
    axes[i].set_xlabel("Easting (m)")
    axes[i].set_ylabel("Northing (m)")
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle("Comparison of Wind Farm Layouts in Next Generation")
plt.tight_layout()
plt.show()