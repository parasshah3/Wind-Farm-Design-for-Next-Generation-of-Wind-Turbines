import farm_design_functions_v6 as fd  # Import the module with your functions
import numpy as np

# Define parameters for the test (all in metres)
v_i = 9.519  # Freestream wind speed (m/s)
Ct = 0.761  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
D = 2 * r_0  # Rotor diameter (m)
z = 100  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# Define turbine positions for testing
turbine_positions_template = {
    1: (0, 0),  # Turbine 1 at (0D, 0D)
    2: (0, 0),  # Placeholder for Turbine 2
}

# Define ranges for downstream (x_ij) and perpendicular (d_ij) distances (in rotor diameters)
x_ij_values = np.linspace(1, 10, 10)  # Downstream distances from 1D to 10D
d_ij_values = np.linspace(0, 3, 10)   # Perpendicular distances from 0D to 3D

# Initialize results tracking
test_results = []

# Test over the ranges of x_ij and d_ij
for x_ij in x_ij_values:
    for d_ij in d_ij_values:
        # Update turbine positions
        turbine_positions = turbine_positions_template.copy()
        turbine_positions[2] = (x_ij * D, d_ij * D)  # Convert distances to metres

        # Use `multiple_wake_model_ij` for Turbine 2
        v_j_xij, _, _, _, _, _, _ = fd.multiple_wake_model_ij(
            v_i, r_0, Ct, x_ij * D, z, z_0, d_ij * D
        )

        # Use `mwm_identical_turbines`
        effective_wind_speeds = fd.mwm_identical_turbines(v_i, Ct, r_0, z, z_0, turbine_positions)

        # Compare results
        try:
            assert abs(effective_wind_speeds[2] - v_j_xij) < 1e-6, f"Mismatch for x_ij={x_ij:.2f}D, d_ij={d_ij:.2f}D"
            test_results.append((x_ij, d_ij, "Passed"))
        except AssertionError as e:
            test_results.append((x_ij, d_ij, f"Failed: {str(e)}"))

# Print test summary
for result in test_results:
    print(f"x_ij={result[0]:.2f}D, d_ij={result[1]:.2f}D -> {result[2]}")

# Final summary
passed_tests = sum(1 for result in test_results if result[2] == "Passed")
total_tests = len(test_results)
print(f"\nSummary: {passed_tests}/{total_tests} tests passed.")