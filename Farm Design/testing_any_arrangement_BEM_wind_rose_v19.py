import matplotlib.pyplot as plt
from farm_design_functions_v12 import energy_yield_with_wind_resource
from farm_design_functions_v11 import transform_turbine_positions

# Inputs
file_path = "/path/to/your/BEM_spreadsheet.xlsm"  # Replace with the actual file path
sheet_name = "WindDist"
Ct = 0.806
r_0 = 126.0
z = 150
z_0 = 7.8662884488e-04
turbine_positions = {
    1: (0, 0),
    2: (4 * r_0 * 2, 2 * r_0),
    3: (4 * r_0 * 2, -2 * r_0),
    4: (8 * r_0 * 2, 0),
    5: (12 * r_0 * 2, 2 * r_0),
    6: (12 * r_0 * 2, -2 * r_0),
    7: (16 * r_0 * 2, 0),
}
wind_rose = {
    0: 0.06,
    30: 0.03,
    60: 0.04,
    90: 0.06,
    120: 0.05,
    150: 0.07,
    180: 0.09,
    210: 0.17,
    240: 0.19,
    270: 0.12,
    300: 0.07,
    330: 0.05,
}

primary_direction = 240  # Primary direction is 240° (Southwest)

# Visualise turbine arrangements for different wind directions
plt.figure(figsize=(16, 12))
for i, direction in enumerate(wind_rose.keys(), start=1):
    # Calculate theta (relative to primary direction)
    theta = (direction - primary_direction) % 360
    if theta > 180:
        theta -= 360  # Ensure theta is in the range [-180, 180]

    # Transform turbine positions
    transformed_positions = transform_turbine_positions(turbine_positions, theta)

    # Plot turbine positions
    plt.subplot(4, 3, i)
    for turbine_id, (x, y) in transformed_positions.items():
        plt.scatter(x / (2 * r_0), y / (2 * r_0), label=f"Turbine {turbine_id}", color="blue")
        plt.annotate(f"{turbine_id}", (x / (2 * r_0), y / (2 * r_0) + 0.2), fontsize=8)

    plt.title(f"Wind Direction: {direction}° (θ = {-theta:+}°)", fontsize=10)
    plt.xlabel("x (Rotor Diameters)")
    plt.ylabel("y (Rotor Diameters)")
    plt.grid()
    plt.tight_layout()

plt.suptitle("Turbine Positions for Different Wind Directions", fontsize=16)
plt.show()

# Calculate annual energy yield
annual_yield = energy_yield_with_wind_resource(
    file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose
)

# Print results
print("\nAnnual Energy Yield (GWhr/year) for each turbine:")
for turbine_id, yield_gwhr in annual_yield.items():
    print(f"Turbine {turbine_id}: {yield_gwhr:.2f} GWhr/year")