import farm_design_functions_v6 as fd  # Import the module with your functions
import numpy as np

# Define parameters for the test (all in metres)
v_i = 9.519  # Freestream wind speed (m/s)
Ct = 0.761  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
D = 2 * r_0  # Rotor diameter (m)
z = 100  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# Define turbine positions based on the given arrangement
turbine_positions = {
    1: (0 * D, 0 * D),    # Turbine 1
    2: (1 * D, 1 * D),    # Turbine 2
    3: (0 * D, -1 * D),   # Turbine 3
    4: (2 * D, 0 * D),    # Turbine 4
    5: (3 * D, 1 * D),    # Turbine 5
    6: (2 * D, -1 * D),   # Turbine 6
    7: (4 * D, 0 * D),    # Turbine 7
}

# Calculate effective wind speeds using `mwm_identical_turbines`
effective_wind_speeds = fd.mwm_identical_turbines(v_i, Ct, r_0, z, z_0, turbine_positions)

# Print results for each turbine 
print("\nResults from `mwm_identical_turbines` for the given arrangement:")
for turbine_id, wind_speed in effective_wind_speeds.items():
    print(f"Turbine {turbine_id}: {wind_speed:.4f} m/s")

# Test for a range of x_ij and d_ij for Turbine 1 -> Turbine 2
print("\nTesting a range of x_ij and d_ij for Turbine 1 -> Turbine 2:")
x_values = np.linspace(1 * D, 5 * D, 5)  # Downstream distances in metres
d_values = np.linspace(0, 2 * D, 5)  # Perpendicular distances in metres

# Iterate through the range of x_ij and d_ij
for x_ij in x_values:
    for d_ij in d_values:
        # Calculate effective wind speed for Turbine 2
        test_turbine_positions = {1: (0, 0), 2: (x_ij, d_ij)}  # Testing two turbines only
        test_results = fd.mwm_identical_turbines(v_i, Ct, r_0, z, z_0, test_turbine_positions)
        
        # Extract and print results for Turbine 2
        v_j = test_results[2]
        print(f"x_ij = {x_ij / D:.2f}D, d_ij = {d_ij / D:.2f}D -> Effective Wind Speed = {v_j:.4f} m/s")