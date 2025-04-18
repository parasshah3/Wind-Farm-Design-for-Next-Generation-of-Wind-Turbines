import xlwings as xw
import matplotlib.pyplot as plt

# File path and sheet name
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# Weibull coefficients for turbines
turbines = [
    {"id": 1, "k": 4.3734, "lambda": 10.2584},
    {"id": 2, "k": 4.3734, "lambda": 9.3256},
    {"id": 3, "k": 4.3734, "lambda": 9.3256},
    {"id": 4, "k": 4.3734, "lambda": 6.3836},
    {"id": 5, "k": 4.3734, "lambda": 6.6977},
    {"id": 6, "k": 4.3734, "lambda": 6.6977},
    {"id": 7, "k": 4.3734, "lambda": 6.4602},
]

# Open the workbook
app = xw.App(visible=False)  # Open Excel in the background
wb = xw.Book(file_path)
sheet = wb.sheets[sheet_name]

# Initialize results list
results = []

# Iterate over turbines
# Iterate over turbines
# Iterate over turbines
for turbine in turbines:
    # Write shape and scale factors into relevant cells
    sheet["G15"].value = turbine["k"]  # Shape factor (k)
    sheet["G16"].value = turbine["lambda"]  # Scale factor (λ)

    # Force recalculation by reading a cell with a formula
    _ = sheet["G4"].value  # Trigger recalculation by accessing a dependent cell

    # Read calculated results
    average_energy_yield = sheet["G4"].value  # Average Energy Yield (MW/hr)
    total_energy_yield = sheet["G7"].value  # Total Energy Yield (GWhr/year)
    capacity_factor = sheet["G10"].value  # Capacity Factor

    # Append results
    results.append({
        "Turbine ID": turbine["id"],
        "Average Energy Yield (MW/hr)": average_energy_yield,
        "Total Energy Yield (GWhr/year)": total_energy_yield,
        "Capacity Factor": capacity_factor
    })

# Close the workbook and Excel app
wb.close()
app.quit()

# Print results to terminal
print("Results:")
for result in results:
    print(result)

# Plot the results
turbine_ids = [r["Turbine ID"] for r in results]
avg_yield = [r["Average Energy Yield (MW/hr)"] for r in results]
total_yield = [r["Total Energy Yield (GWhr/year)"] for r in results]
capacity_factors = [r["Capacity Factor"] for r in results]

plt.figure(figsize=(12, 6))

# Plot average energy yield
plt.subplot(1, 3, 1)
plt.bar(turbine_ids, avg_yield, color="skyblue")
plt.title("Average Energy Yield (MW/hr)")
plt.xlabel("Turbine ID")
plt.ylabel("Energy Yield (MW/hr)")

# Plot total energy yield
plt.subplot(1, 3, 2)
plt.bar(turbine_ids, total_yield, color="orange")
plt.title("Total Energy Yield (GWhr/year)")
plt.xlabel("Turbine ID")
plt.ylabel("Energy Yield (GWhr/year)")

# Plot capacity factor
plt.subplot(1, 3, 3)
plt.bar(turbine_ids, capacity_factors, color="green")
plt.title("Capacity Factor")
plt.xlabel("Turbine ID")
plt.ylabel("Capacity Factor")

plt.tight_layout()
plt.show()