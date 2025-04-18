import matplotlib.pyplot as plt
import pandas as pd
from farm_design_functions_v28 import energy_yield_with_wind_resource

# Define file path and sheet name
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/25MW_EighthIteration.xlsm"
sheet_name = "WindDist"

# Define wind turbine parameters
Ct = 0.790
r_0 = 124.7
z = 140
z_0 = 7.8662884488e-04

# Load wind turbine UTM positions from CSV
file_path_csv = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Turbine positions UTM.csv"
df = pd.read_csv(file_path_csv, header=None, names=["Easting", "Northing"])

# Store wind turbine positions in a dictionary
turbine_positions = {i+1: (row["Easting"], row["Northing"]) for i, row in df.iterrows()}

# Define the wind rose with frequencies
wind_rose = {
    0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
    180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
}
primary_direction = 240  # Most common wind direction

# Compute the annual energy yield for each turbine
annual_yield = energy_yield_with_wind_resource(
    file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose, primary_direction
)

# Compute the total energy yield for the entire wind farm
total_farm_yield = sum(annual_yield.values())

# Print results
print("\nAnnual Energy Yield (GWhr/year) for each turbine:")
for turbine_id, yield_gwhr in annual_yield.items():
    print(f"Turbine {turbine_id}: {yield_gwhr:.2f} GWhr/year")

print(f"\nTotal Annual Energy Yield for Wind Farm: {total_farm_yield:.2f} GWhr/year")

# Plot annual energy yield for each turbine
import matplotlib.ticker as ticker

plt.figure(figsize=(15, 6))  # adjust width and height

# Bar chart
plt.bar(annual_yield.keys(), annual_yield.values())  # default color

# Labels and title
plt.xlabel("Turbine ID", fontsize=14, fontname="Arial")
plt.ylabel("Annual Energy Yield (GWhr/year)", fontsize=14, fontname="Arial")
#plt.title("Annual Energy Yield per Turbine", fontsize=16, fontweight='bold's, fontname="Arial")

# Axis ticks
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")

# Set x and y axis intervals (optional – example below)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))  # every 10 turbines
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(20))  # every 10 GWh

# Grid styling
plt.grid(axis="y",  linewidth=0.7, alpha=0.7)

# Tight layout for better spacing
plt.tight_layout()

plt.show()