

import matplotlib.pyplot as plt
from math import radians

import numpy as np
import matplotlib.pyplot as plt
from farm_design_functions_v25 import generate_parallelogram_grid, define_parameter_space

# Sheringham Shoal UTM boundary
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # close loop
])

anchor = (373690.373, 5887454.043)

# Boundary side lengths and angle (inferred or computed earlier)
A = 7529.81  # m (along rows)
B = 6350.15  # m (along columns)
alpha_deg = 133.36
rotor_diameter = 249.4

# Define search space
param_space = define_parameter_space(rotor_diameter, A, B, alpha_deg,
                                     a_steps=10, b_steps=10, theta_steps=)

for idx, (a, b, theta) in enumerate(param_space):
    layout = generate_parallelogram_grid(a, b, theta, anchor, max_extent=20000)

    # Extract turbine coordinates
    coords = np.array(list(layout.values()))

    # Optional: translate grid centroid to boundary centroid
    grid_centroid = np.mean(coords, axis=0)
    boundary_centroid = np.mean(boundary_utm[:-1], axis=0)  # exclude repeated last point
    offset = boundary_centroid - grid_centroid
    coords_translated = coords + offset

    # Plot
    plt.figure(figsize=(7, 7))
    plt.plot(boundary_utm[:, 0], boundary_utm[:, 1], 'k-', label="Site boundary")
    plt.scatter(coords_translated[:, 0], coords_translated[:, 1], c='r', s=15, label="Turbines")
    
    # Ensure full boundary + layout is visible
    all_x = np.concatenate([boundary_utm[:, 0], coords_translated[:, 0]])
    all_y = np.concatenate([boundary_utm[:, 1], coords_translated[:, 1]])
    x_pad = 1000
    y_pad = 1000
    plt.xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    plt.ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    plt.title(f"Grid #{idx+1}: a={a:.1f}m, b={b:.1f}m, θ={theta:.1f}°")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()