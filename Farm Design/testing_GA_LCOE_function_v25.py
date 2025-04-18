import numpy as np
from farm_design_functions_v14 import generate_random_layout, LCOE

# Define file path to BEM spreadsheet
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/20MW_ThirdIteration- Wind Farm.xlsm"
sheet_name = "WindDist"

# Define wind turbine parameters
Ct = 0.806  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
z = 150  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# Define boundary UTM coordinates
boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Closing the boundary loop
])

# Define spacing constraints and turbine count range
rotor_diameter = 126.0
min_spacing = 3 * rotor_diameter  # Minimum spacing (3D)
num_turbines_range = (50, 150)  # Number of turbines (to optimize)

# Wind Rose Data (Wind direction frequencies)
wind_rose = {
    0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
    180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
}
primary_direction = 240  # Most common wind direction

# Generate a random wind farm layout
random_layout = generate_random_layout(boundary_utm, min_spacing, num_turbines_range)

# Compute LCOE for the generated layout using wind resource data
lcoe_value = LCOE(file_path, sheet_name, Ct, r_0, z, z_0, random_layout, wind_rose, primary_direction)

# Print results
print("\n--- LCOE Calculation Results ---")
print(f"Number of turbines: {len(random_layout)}")
print(f"Computed LCOE: {lcoe_value:.2f} £/MWhr")