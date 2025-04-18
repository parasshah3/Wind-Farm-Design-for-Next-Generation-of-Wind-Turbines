import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon

from farm_design_functions_v25 import generate_regular_arrangement, define_parameter_space

# Sheringham Shoal UTM boundary
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # close loop
])

# Anchor point (origin) at vertex 4
anchor = (373690.373, 5887454.043)

# Boundary side lengths and angle
A = 7529.81  # Row-aligned boundary length
B = 6350.15  # Column-aligned boundary length
alpha_deg = 133.36
rotor_diameter = 249.4

# Define search space
param_space = define_parameter_space(
    rotor_diameter, A, B, alpha_deg,
    a_steps=10, b_steps=10, theta_steps=15  # <- Adjust for quick test
)

# Plot each layout
for idx, (a, b, theta) in enumerate(param_space):
    layout = generate_regular_arrangement(a, b, theta, A, B, alpha_deg)

    # Skip if layout is empty
    if not layout:
        continue

    coords = np.array(list(layout.values()))
    ids = list(layout.keys())

    # Plotting
    plt.figure(figsize=(7, 7))
    plt.plot(boundary_utm[:, 0], boundary_utm[:, 1], 'k-', label="Site boundary")
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=10, label="Turbines")

    # Annotate each turbine with ID
    for tid, (x, y) in zip(ids, coords):
        plt.text(x, y + 25, str(tid), fontsize=6, ha='center', color='black')

    # Ensure axes cover full layout
    all_x = np.concatenate([boundary_utm[:, 0], coords[:, 0]])
    all_y = np.concatenate([boundary_utm[:, 1], coords[:, 1]])
    x_pad, y_pad = 1000, 1000
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