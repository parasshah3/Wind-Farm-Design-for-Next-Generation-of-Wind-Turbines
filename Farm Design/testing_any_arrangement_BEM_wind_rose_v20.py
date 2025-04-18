import matplotlib.pyplot as plt
from farm_design_functions_v12 import energy_yield_with_wind_resource
from farm_design_functions_v11 import transform_turbine_positions

# File Path and Sheet Name
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# Wind Turbine Arrangement (x, y in metres)
r_0 = 126.0  # Rotor radius (m)
Ct = 0.806  # Thrust coefficient
z = 150  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

turbine_positions = {
    1: (0, 0),
    2: (4 * r_0 * 2, 2 * r_0),
    3: (4 * r_0 * 2, -2 * r_0),
    4: (8 * r_0 * 2, 0),
    5: (12 * r_0 * 2, 2 * r_0),
    6: (12 * r_0 * 2, -2 * r_0),
    7: (16 * r_0 * 2, 0),
}

# Wind Rose (Wind Direction Frequency)
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

primary_direction = 240  # Most common wind direction at site (Southwest)

# Compute the Annual Energy Yield for each turbine
annual_yield = energy_yield_with_wind_resource(
    file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose
)

# Print results
print("\nAnnual Energy Yield (GWhr/year) for each turbine:")
for turbine_id, yield_gwhr in annual_yield.items():
    print(f"Turbine {turbine_id}: {yield_gwhr:.2f} GWhr/year")

# Extract data for plotting
turbine_ids = list(annual_yield.keys())
energy_yields = list(annual_yield.values())

# **Plot 1: Bar Chart of Annual Energy Yield**
plt.figure(figsize=(10, 6))
plt.bar(turbine_ids, energy_yields, color="skyblue", edgecolor="black")
plt.xlabel("Turbine ID")
plt.ylabel("Annual Energy Yield (GWhr/year)")
plt.title("Annual Energy Yield for Each Turbine")
plt.xticks(turbine_ids)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# **Plot 2: Stacked Bar Chart - Contribution from Wind Directions**
wind_direction_contributions = {tid: [] for tid in turbine_ids}
wind_directions = list(wind_rose.keys())

for wind_dir in wind_directions:
    theta = -(wind_dir - primary_direction) % 360
    if theta > 180:
        theta += 360  # Ensure theta is within [-180, 180]

    transformed_positions = transform_turbine_positions(turbine_positions, theta)
    energy_contributions = energy_yield_with_wind_resource(
        file_path, sheet_name, Ct, r_0, z, z_0, transformed_positions, {wind_dir: wind_rose[wind_dir]}
    )

    for tid in turbine_ids:
        wind_direction_contributions[tid].append(energy_contributions.get(tid, 0))

# Plot Stacked Bar Chart
plt.figure(figsize=(12, 7))
bottom_values = [0] * len(turbine_ids)
colors = plt.cm.viridis_r(np.linspace(0, 1, len(wind_directions)))

for i, wind_dir in enumerate(wind_directions):
    contrib = [wind_direction_contributions[tid][i] for tid in turbine_ids]
    plt.bar(
        turbine_ids, contrib, bottom=bottom_values, color=colors[i], edgecolor="black", label=f"{wind_dir}°"
    )
    bottom_values = [bottom_values[j] + contrib[j] for j in range(len(turbine_ids))]

plt.xlabel("Turbine ID")
plt.ylabel("Annual Energy Yield (GWhr/year)")
plt.title("Energy Yield Contribution by Wind Direction")
plt.xticks(turbine_ids)
plt.legend(title="Wind Direction (°)", loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()