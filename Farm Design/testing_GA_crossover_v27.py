import matplotlib.pyplot as plt
import numpy as np
from farm_design_functions_v15 import generate_random_layout, random_crossover, initialise_population

# Define Sheringham Shoal wind farm boundary (UTM coordinates)
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Closing the loop
])

# Define turbine parameters
rotor_diameter = 126.0  # Rotor diameter in meters
min_spacing = 3 * rotor_diameter  # Minimum spacing (3D)
population_size = 10  # Number of layouts to generate
num_turbines_range = (5, 50)  # Range of turbine counts

# Generate population of layouts
population = initialise_population(population_size, boundary_utm, min_spacing, num_turbines_range)

# Select two random parents
parent_1 = population[0]  # Fitter parent (lower LCOE in actual GA)
parent_2 = population[1]  # Less fit parent (higher LCOE in actual GA)

# Perform crossover
child_layout, crossover_region = random_crossover(parent_1, parent_2, boundary_utm, min_spacing)

# Plot function
def plot_wind_farm(ax, layout, boundary, region, title, color):
    """Plot wind turbine layout with boundary and crossover region."""
    ax.plot(boundary[:, 0], boundary[:, 1], 'b-', label="Boundary")
    ax.fill(*region.exterior.xy, color='gray', alpha=0.4, label="Crossover Region")

    x, y = zip(*layout.values())
    ax.scatter(x, y, c=color, marker='o', label=f"Turbines ({len(layout)})")
    
    for tid, (tx, ty) in layout.items():
        ax.text(tx, ty, str(tid), fontsize=8, ha="center", color="black")
    
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid()

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot_wind_farm(axes[0], parent_1, boundary_utm, crossover_region, "Parent 1 (Fitter Layout)", 'red')
plot_wind_farm(axes[1], parent_2, boundary_utm, crossover_region, "Parent 2 (Less Fit Layout)", 'blue')
plot_wind_farm(axes[2], child_layout, boundary_utm, crossover_region, "Child Layout (Crossover Result)", 'green')

plt.tight_layout()
plt.show()