import farm_design_functions_v8 as fd  # Import the module with your functions
import numpy as np
import pandas as pd

# Define parameters for the test (all in metres)
Ct = 0.806  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
D = 2 * r_0  # Rotor diameter (m)
z = 140  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# Test 1: Original arrangement with 7 turbines
A = 1.5 * D  # Scaling factor for turbine spacing
turbine_positions_test1 = {
    1: (0 * A, 0 * A),
    2: (1 * A, 0.5 * A),
    3: (1 * A, -0.5 * A),
    4: (2 * A, 0 * A),
    5: (3 * A, 0.5 * A),
    6: (3 * A, -0.5 * A),
    7: (4 * A, 0 * A),
}

def run_test_and_save_csv(turbine_positions, output_path, diameter):
    """
    Run the multiple wake model test and save results to a CSV file.

    Parameters:
    turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions
    output_path (str): Full path to the output CSV file
    diameter (float): Rotor diameter (m) to scale coordinates to diameters
    """
    # Calculate Weibull parameters using `mwm_identical_turbines_WD`
    results = fd.mwm_identical_turbines_WD(Ct, r_0, z, z_0, turbine_positions)

    # Prepare data for saving
    data = []
    for turbine_id, data_dict in results.items():
        x, y = turbine_positions[turbine_id]
        x_diameters = x / diameter
        y_diameters = y / diameter
        k = data_dict['k']
        lambda_ = data_dict['lambda']
        data.append({'Turbine ID': turbine_id, 'x (D)': x_diameters, 'y (D)': y_diameters, 'k': k, 'λ': lambda_})

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

# Specify output path
output_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/Test_arrangement_1.csv"

# Run Test 1 and save results
run_test_and_save_csv(turbine_positions_test1, output_path, D)