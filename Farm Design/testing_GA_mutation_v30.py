import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from farm_design_functions_v17 import generate_random_layout, mutate_layout

# Define wind farm boundary (UTM coordinates)
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Closing the boundary loop
])

# Define turbine parameters
rotor_diameter = 124.7  # Rotor diameter in meters
min_spacing = 2 * 2 * rotor_diameter  # Minimum spacing (3D)
num_turbines_range = (40, 110)  # Range of turbine counts

# Generate an initial turbine layout
print("Generating initial layout...")
initial_layout = generate_random_layout(boundary_utm, min_spacing, num_turbines_range)

# Apply mutation to the layout
print("Applying mutation...")
mutated_layout = mutate_layout(initial_layout, boundary_utm, min_spacing, rotor_diameter)

# Convert boundary to a polygon for plotting
boundary_polygon = Polygon(boundary_utm)

# **Plot results**
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# **Original Layout**
ax[0].plot(*boundary_polygon.exterior.xy, 'b-', label="Boundary")
original_x, original_y = zip(*initial_layout.values())
ax[0].scatter(original_x, original_y, c='r', marker='o', label=f"Original Turbines ({len(initial_layout)})")
ax[0].set_title("Original Wind Farm Layout")
ax[0].set_xlabel("Easting (m)")
ax[0].set_ylabel("Northing (m)")
ax[0].legend()
ax[0].grid()

# **Mutated Layout**
ax[1].plot(*boundary_polygon.exterior.xy, 'b-', label="Boundary")
mutated_x, mutated_y = zip(*mutated_layout.values())
ax[1].scatter(mutated_x, mutated_y, c='g', marker='o', label=f"Mutated Turbines ({len(mutated_layout)})")
ax[1].set_title("Mutated Wind Farm Layout")
ax[1].set_xlabel("Easting (m)")
ax[1].set_ylabel("Northing (m)")
ax[1].legend()
ax[1].grid()

plt.suptitle("Testing Mutation Function - Wind Farm Layout")
plt.show()

# **Debugging Output**
print(f"Original number of turbines: {len(initial_layout)}")
print(f"Mutated number of turbines: {len(mutated_layout)}")