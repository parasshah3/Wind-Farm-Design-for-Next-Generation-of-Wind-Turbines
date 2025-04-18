import numpy as np
import matplotlib.pyplot as plt
from farm_design_functions_v14 import initialise_population

# Define wind farm boundary (UTM coordinates)
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Close loop
])

# Turbine parameters
rotor_diameter = 124.7 * 2
min_spacing = 3 * rotor_diameter
population_size = 10
num_turbines_range = (15, 50)

# Generate layouts
population = initialise_population(population_size, boundary_utm, min_spacing, num_turbines_range)

print("\nInitial Population Layouts:")
#print only first 2 layouts
for i in range(2):
    print(f"Number of turbines: {len(population[i])}")
    print(f"Layout {i + 1}: {population[i]}")
    print(" ")


# Plot style
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16
})

# Plot two layouts
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for i in range(2):
    ax = axes[i]
    layout = population[i]
    coords = np.array(list(layout.values()))
    ids = list(layout.keys())

    # Boundary and turbine scatter
    ax.plot(boundary_utm[:, 0], boundary_utm[:, 1], color='black', linewidth=0.85, alpha=0.7, label="Wind Farm Boundary")
    ax.scatter(coords[:, 0], coords[:, 1], s=25, label=f"Turbines ({len(layout)})")

    # Annotate turbines
    for tid, (x, y) in layout.items():
        ax.text(x, y + 70, str(tid), fontsize=10, ha='center', color='black', fontname="Arial")

    # Padding and limits
    all_x = np.concatenate([boundary_utm[:, 0], coords[:, 0]])
    all_y = np.concatenate([boundary_utm[:, 1], coords[:, 1]])
    x_pad, y_pad = 1000, 1000
    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    # Labels and grid
    ax.set_xlabel("Easting (m)", fontsize=16)
    ax.set_ylabel("Northing (m)", fontsize=16)
    #ax.set_title(f"Layout {i+1}", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='-', linewidth=0.7, alpha=0.7)
    ax.legend(prop={"family": "Arial", "size": 16})
    ax.tick_params(axis='both', labelsize=14)
    ax.axis("equal")

    # Scientific offset formatting
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(-3, 4), useMathText=True)
    ax.xaxis.offsetText.set_fontsize(10)
    ax.yaxis.offsetText.set_fontsize(10)
    ax.xaxis.offsetText.set_fontname("Arial")
    ax.yaxis.offsetText.set_fontname("Arial")

plt.tight_layout()
plt.show()