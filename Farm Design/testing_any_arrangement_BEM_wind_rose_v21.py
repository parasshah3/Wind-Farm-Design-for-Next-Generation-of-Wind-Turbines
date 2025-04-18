import matplotlib.pyplot as plt
from farm_design_functions_v12 import energy_yield_with_wind_resource
from farm_design_functions_v11 import transform_turbine_positions
from farm_design_functions_v12 import mwm_identical_turbines_BEM_any_direction

# Inputs
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"
Ct = 0.806
r_0 = 124.7
z = 150
z_0 = 7.8662884488e-04

# Turbine arrangement (line perpendicular to primary wind direction, 240°)
turbine_positions = {
    1: (0, 0),
    2: (0, 2 * r_0 * 2),
    3: (0, 4 * r_0 * 2),
    4: (0, 6 * r_0 * 2),
    5: (0, 8 * r_0 * 2),
    6: (0, 10 * r_0 * 2),
    7: (0, 12 * r_0 * 2),
}

# Wind resource (wind rose) with frequencies
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

# Primary direction
primary_direction = 240



# Case 1: Using wind resource (wind rose)
annual_yield_resource = energy_yield_with_wind_resource(
    file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose
)

# Case 2: Constant southeastern wind direction
annual_yield_southeast = mwm_identical_turbines_BEM_any_direction(
    file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, theta=0
)

# Plot energy yield comparison
turbine_ids = list(annual_yield_resource.keys())
yield_resource = list(annual_yield_resource.values())
yield_southeast = [v["Total Energy Yield (GWhr/year)"] for v in annual_yield_southeast.values()]

plt.figure(figsize=(10, 6))
plt.bar(
    [turbine_id - 0.2 for turbine_id in turbine_ids],
    yield_resource,
    width=0.4,
    label="Wind Resource (Wind Rose)",
    color="skyblue",
)
plt.bar(
    [turbine_id + 0.2 for turbine_id in turbine_ids],
    yield_southeast,
    width=0.4,
    label="Constant Southeastern Wind (240°)",
    color="orange",
)
plt.xticks(turbine_ids)
plt.xlabel("Turbine ID")
plt.ylabel("Annual Energy Yield (GWhr/year)")
plt.title("Comparison of Annual Energy Yield for Different Wind Scenarios")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Print results
print("Annual Energy Yield Comparison:")
print(f"{'Turbine ID':<12}{'Wind Resource (GWhr)':<25}{'Southeastern Wind (GWhr)':<30}")
for turbine_id in turbine_ids:
    print(
        f"{turbine_id:<12}{yield_resource[turbine_id - 1]:<25.2f}{yield_southeast[turbine_id - 1]:<30.2f}"
    )