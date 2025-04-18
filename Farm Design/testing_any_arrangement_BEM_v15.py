from farm_design_functions_v9 import mwm_identical_turbines_BEM
import matplotlib.pyplot as plt

# File path and sheet name
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# Define turbine arrangement and parameters
Ct = 0.806  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
z = 140  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)
A = 6 * (2 * r_0)  # Spacing in multiples of rotor diameter
turbine_positions = {
    1: (0 * A, 0 * A),
    2: (1 * A, 0.5 * A),
    3: (1 * A, -0.5 * A),
    4: (2 * A, 0 * A),
    5: (3 * A, 0.5 * A),
    6: (3 * A, -0.5 * A),
    7: (4 * A, 0 * A),
}

# Run the function and get results
results = mwm_identical_turbines_BEM(file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions)

# Print results
print("\n--- Results ---")
for turbine_id, metrics in results.items():
    print(f"Turbine {turbine_id}: {metrics}")

# Extract data for plotting
turbine_ids = list(results.keys())
average_energy_yields = [metrics["Average Energy Yield (MW/hr)"] for metrics in results.values()]
total_energy_yields = [metrics["Total Energy Yield (GWhr/year)"] for metrics in results.values()]
capacity_factors = [metrics["Capacity Factor"] for metrics in results.values()]

# Plot results
plt.figure(figsize=(15, 5))

# Average Energy Yield
plt.subplot(1, 3, 1)
plt.bar(turbine_ids, average_energy_yields, color="skyblue")
plt.title("Average Energy Yield (MW/hr)")
plt.xlabel("Turbine ID")
plt.ylabel("Energy Yield (MW/hr)")

# Total Energy Yield
plt.subplot(1, 3, 2)
plt.bar(turbine_ids, total_energy_yields, color="orange")
plt.title("Total Energy Yield (GWhr/year)")
plt.xlabel("Turbine ID")
plt.ylabel("Energy Yield (GWhr/year)")

# Capacity Factor
plt.subplot(1, 3, 3)
plt.bar(turbine_ids, capacity_factors, color="green")
plt.title("Capacity Factor")
plt.xlabel("Turbine ID")
plt.ylabel("Capacity Factor")

plt.tight_layout()
plt.show()