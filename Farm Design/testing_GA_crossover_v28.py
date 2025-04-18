import numpy as np
import matplotlib.pyplot as plt
from farm_design_functions_v16 import generate_random_layout, random_crossover, initialise_population

# Sheringham Shoal boundary (UTM)
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)
])

# Parameters
rotor_diameter = 124.7
min_spacing = 3 * rotor_diameter
population = initialise_population(10, boundary_utm, min_spacing, (20, 50))

# Select parents and perform crossover
parent_1 = population[0]
parent_2 = population[1]
child_layout, crossover_region = random_crossover(parent_1, parent_2, boundary_utm, min_spacing)

# Plotting function
def plot_wind_farm(ax, layout, boundary, region, title, color,
                   show_region_label=False, show_boundary_label=False,
                   show_xlabel=True, show_ylabel=True):
    coords = np.array(list(layout.values()))

    # Boundary
    ax.plot(boundary[:, 0], boundary[:, 1], color='black', linewidth=0.85, alpha=0.7,
            label="Wind Farm Boundary" if show_boundary_label else None)

    # Crossover region
    ax.fill(*region.exterior.xy, color='gray', alpha=0.4,
            label="Crossover Region" if show_region_label else None)

    # Turbines
    ax.scatter(coords[:, 0], coords[:, 1], s=35, color=color, label=f"Turbines ({len(layout)})")

    # Axis limits and formatting
    all_x = np.concatenate([boundary[:, 0], coords[:, 0]])
    all_y = np.concatenate([boundary[:, 1], coords[:, 1]])
    ax.set_xlim(all_x.min() - 1000, all_x.max() + 1000)
    ax.set_ylim(all_y.min() - 1000, all_y.max() + 1000)

    if show_xlabel:
        ax.set_xlabel("Easting (m)", fontsize=16, fontname="Arial")
    if show_ylabel:
        ax.set_ylabel("Northing (m)", fontsize=16, fontname="Arial")

    ax.set_title(title, fontsize=16, fontname="Arial")
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, linestyle='-', linewidth=0.7, alpha=0.7)
    ax.axis("equal")

    ax.ticklabel_format(style='scientific', axis='both', scilimits=(-3, 4), useMathText=True)
    ax.xaxis.offsetText.set_fontsize(10)
    ax.yaxis.offsetText.set_fontsize(10)
    ax.xaxis.offsetText.set_fontname("Arial")
    ax.yaxis.offsetText.set_fontname("Arial")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles=handles, prop={"family": "Arial", "size": 14})

# Create 3-panel figure
plt.rcParams.update({"font.family": "Arial", "font.size": 16})
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plot_wind_farm(
    axes[0], parent_1, boundary_utm, crossover_region,
    title="Parent 1 (Fitter Layout)", color="blue",
    show_region_label=True, show_boundary_label=True,
    show_xlabel=True, show_ylabel=True
)

plot_wind_farm(
    axes[1], parent_2, boundary_utm, crossover_region,
    title="Parent 2 (Less Fit Layout)", color='red',
    show_xlabel=False, show_ylabel=False
)

plot_wind_farm(
    axes[2], child_layout, boundary_utm, crossover_region,
    title="Child Layout", color='green',
    show_xlabel=False, show_ylabel=False
)

plt.tight_layout()
plt.show()