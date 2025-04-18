import numpy as np
from farm_design_functions_v15 import initialise_population, reproduction_selection, LCOE

# Define wind farm boundary (UTM coordinates)
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Closing the boundary loop
])

# Define wind turbine parameters
Ct = 0.806  
r_0 = 126.0  
z = 150  
z_0 = 7.8662884488e-04  

# Define genetic algorithm parameters
population_size = 10  
num_turbines_range = (2, 13)  
min_spacing = 3 * r_0  

# Define wind rose (wind direction frequencies)
wind_rose = {
    0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
    180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
}
primary_direction = 240  

# Define file path to BEM spreadsheet
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# Generate initial population
population = initialise_population(population_size, boundary_utm, min_spacing, num_turbines_range)

# Perform selection
best_parents = reproduction_selection(population, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose)

# Print results
print(f"Selected {len(best_parents)} best parents.")
for i, layout in enumerate(best_parents):
    print(f"Parent {i+1}: {len(layout)} turbines, LCOE = {LCOE(file_path, sheet_name, Ct, r_0, z, z_0, layout, wind_rose):.2f} £/MWhr")