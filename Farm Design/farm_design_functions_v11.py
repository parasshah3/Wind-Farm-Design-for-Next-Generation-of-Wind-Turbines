import numpy as np
from math import pi
from estimating_weibull_v7_any_height import calculate_weibull_params_vj,calculate_mean_wind_speed,calculate_std_dev  # Import Weibull calculations

def jensen_single_model(v0, Ct, r0, x, z, z0):
    """
    Calculate the reduced wind speed downstream in the wake zone using Jensen's single wake model.
    """
    # Compute the decay constant (alpha)
    alpha = 1 / (2 * np.log(z / z0))
    # print(f"alpha = {alpha}")

    # Calculate the wake radius at downstream distance x
    rx = r0 + alpha * x

    # Calculate reduced wind speed in the wake
    reduced_speed = v0 + v0 * (np.sqrt(1 - Ct) - 1) * (r0 / rx) ** 2
    return reduced_speed


def multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij):
    """
    Calculate the effective wind speed at turbine j due to the wake from turbine i,
    and return intermediate parameters.
    """
    # Initialize shadowing flags
    partial_shadowing = False
    complete_shadowing = False
    no_shadowing = False

    # Default values for L_ij, z_ij, A_shadow_i
    L_ij = None
    z_ij = None
    A_shadow_i = None

    # Compute the decay constant (alpha)
    alpha = 0.5 / np.log(z / z_0)  # Decay constant

    # Calculate the wake radius at downstream distance x_ij
    r_i_xij = r_0 + alpha * x_ij  # Wake radius at distance x_ij
    #print(f"r_i_xij (wake radius): {r_i_xij}")

    # Calculate the rotor swept area of turbine j
    A_0 = np.pi * r_0**2

    # Case 1: Complete shadowing (entire rotor within the wake)
    if d_ij + r_0 <= r_i_xij:
        A_shadow_i = A_0  # Entire rotor is shadowed
        complete_shadowing = True
        print("Case: Complete shadowing")
        v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij) ** 2)
        return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i

    # Case 2: No shadowing (no overlap between wake and turbine rotor)
    elif d_ij >= (r_0 + r_i_xij):
        A_shadow_i = 0  # No shadowing occurs
        no_shadowing = True
        print("Case: No shadowing")
        return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i

    # Case 3: Partial shadowing (partial overlap)
    else:
        print("Case: Partial shadowing")
        # Calculate the height of the shadowed area (z_ij)
        term_1 = 4 * d_ij**2 * r_i_xij**2
        term_2 = (d_ij**2 - r_0**2 + r_i_xij**2) ** 2
        z_ij = (1 / d_ij) * np.sqrt(term_1 - term_2) if term_1 > term_2 else 0
        #print(f"z_ij (height of shadowed area): {z_ij}")

        # Calculate the perpendicular distance L_ij
        L_ij = abs(d_ij - np.sqrt(r_0**2 - (z_ij / 2) ** 2)) if z_ij > 0 else 0
        #print(f"L_ij (perpendicular distance): {L_ij}")

        # Calculate the shadowed area A_shadow_i
        if L_ij > 0:
            try:
                A_shadow_i = (
                    r_0**2 * np.arccos((d_ij**2 + r_0**2 - r_i_xij**2) / (2 * d_ij * r_0))
                    + r_i_xij**2 * np.arccos((d_ij**2 + r_i_xij**2 - r_0**2) / (2 * d_ij * r_i_xij))
                    - 0.5 * d_ij * z_ij
                )
                partial_shadowing = True
                print(f"A_shadow_i/A_0 (normalised shadowed area): {A_shadow_i/A_0}")
                v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij) ** 2 * (A_shadow_i / A_0))
                return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i
            except ValueError:
                A_shadow_i = 0  # Handle invalid geometry
        else:
            A_shadow_i = 0  # No valid overlap

    # Default return for no valid overlap or partial shadowing calculation errors
    no_shadowing = True
    print("Case: No valid overlap (default case)")
    return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i


def mwm_identical_turbines(v_0, Ct, r_0, z, z_0, turbine_positions):
    """
    Multiple Wake Model for a Wind Farm of Identical Turbines (Using multiple_wake_model_ij).
    """
    # Sort turbines by upstream position (x-coordinate)
    sorted_turbines = sorted(turbine_positions.items(), key=lambda t: t[1][0])  # Sort by x-coordinate
    print(f"Sorted turbines by x-coordinate: {sorted_turbines}")

    effective_wind_speeds = {}  # To store results

    for j, (j_id, (x_j, y_j)) in enumerate(sorted_turbines):
        total_deficit_squared = 0  # Use quadratic summation of wake deficits
        print(f"Considering turbine {j_id} at position ({x_j}, {y_j})")

        for i, (i_id, (x_i, y_i)) in enumerate(sorted_turbines[:j]):  # Only consider upstream turbines
            x_ij = x_j - x_i  # Downstream distance
            #print(f"x_ij (downstream distance): {x_ij}")
            d_ij = abs(y_j - y_i)  # Perpendicular distance
            #print(f"d_ij (perpendicular distance): {d_ij}")

            if x_ij > 0:  # Only consider turbines upstream
                # Use the effective wind speed at turbine i
                v_i = effective_wind_speeds.get(i_id, v_0)

                print(f"Considering turbine pair; {i_id} -> {j_id}")

                # Calculate wake effect using multiple_wake_model_ij
                v_j_xij, _, _, _, _, _, _ = multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij)
                print(f"v_j_xij (wind speed at j due to i): {v_j_xij}")

                # Calculate wake deficit for turbine j from turbine i
                wake_deficit = (v_i - v_j_xij) / v_0  # Normalized wake deficit
                print(f"wake_deficit: {wake_deficit}")
                total_deficit_squared += wake_deficit**2  # Quadratic summation

        # Combine wake deficits using energy superposition
        total_deficit = np.sqrt(total_deficit_squared)
        effective_wind_speeds[j_id] = v_0 * (1 - total_deficit)
        print(f"Turbine {j_id} effective wind speed: {effective_wind_speeds[j_id]}")
        print(" ")

    return effective_wind_speeds

def mwm_identical_turbines_speed_sd(v_0, sigma_0, Ct, r_0, z, z_0, turbine_positions):
    """
    Multiple Wake Model for a Wind Farm of Identical Turbines (Using multiple_wake_model_ij),
    including standard deviation scaling.

    Parameters:
    v_0 (float): Freestream mean wind velocity (m/s)
    sigma_0 (float): Freestream wind speed standard deviation (m/s)
    Ct (float): Thrust coefficient
    r_0 (float): Rotor radius (m)
    z (float): Hub height (m)
    z_0 (float): Surface roughness length (m)
    turbine_positions (dict): Dictionary with keys as turbine IDs and values as (x, y) coordinates

    Returns:
    dict: Effective mean wind speeds and standard deviations at each turbine
          (key: turbine ID, value: {'v_j': mean wind speed, 'sigma_j': standard deviation})
    """
    # Sort turbines by upstream position (x-coordinate)
    sorted_turbines = sorted(turbine_positions.items(), key=lambda t: t[1][0])  # Sort by x-coordinate
    print(f"Sorted turbines by x-coordinate: {sorted_turbines}")

    results = {}  # To store effective wind speeds and standard deviations

    for j, (j_id, (x_j, y_j)) in enumerate(sorted_turbines):
        total_deficit_squared = 0  # Use quadratic summation of wake deficits
        print(f"Considering turbine {j_id} at position ({x_j}, {y_j})")

        for i, (i_id, (x_i, y_i)) in enumerate(sorted_turbines[:j]):  # Only consider upstream turbines
            x_ij = x_j - x_i  # Downstream distance
            d_ij = abs(y_j - y_i)  # Perpendicular distance

            if x_ij > 0:  # Only consider turbines upstream
                # Use the effective wind speed at turbine i
                v_i = results.get(i_id, {}).get('v_j', v_0)

                print(f"Considering turbine pair; {i_id} -> {j_id}")

                # Calculate wake effect using multiple_wake_model_ij
                v_j_xij, _, _, _, _, _, _ = multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij)
                print(f"v_j_xij (wind speed at j due to i): {v_j_xij}")

                # Calculate wake deficit for turbine j from turbine i
                wake_deficit = (v_i - v_j_xij) / v_0  # Normalized wake deficit
                print(f"wake_deficit: {wake_deficit}")
                total_deficit_squared += wake_deficit**2  # Quadratic summation

        # Combine wake deficits using energy superposition
        total_deficit = np.sqrt(total_deficit_squared)
        v_j = v_0 * (1 - total_deficit)  # Effective mean wind speed

        # Scale the standard deviation based on the mean wind speed
        sigma_j = (v_j / v_0) * sigma_0  # Standard deviation scaling
        print(f"Turbine {j_id} effective wind speed: {v_j}, standard deviation: {sigma_j}")
        print(" ")

        # Store results for the turbine
        results[j_id] = {'v_j': v_j, 'sigma_j': sigma_j}

    return results

def mwm_identical_turbines_WD(Ct, r_0, z, z_0, turbine_positions):
    """
    Multiple Wake Model with Weibull Distribution Scaling.

    Parameters:
    - Ct (float): Thrust coefficient
    - r_0 (float): Rotor radius (m)
    - z (float): Hub height (m)
    - z_0 (float): Surface roughness length (m)
    - turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions

    Returns:
    - dict: Contains effective wind speeds, standard deviations, shape factors, and scale factors for each turbine.
    """
    # Compute freestream wind speed and standard deviation at height z
    v_0 = calculate_mean_wind_speed(z)  # Get v_0 dynamically
    sigma_0 = calculate_std_dev(z)      # Get σ_0 dynamically

    # Sort turbines by upstream position (x-coordinate)
    sorted_turbines = sorted(turbine_positions.items(), key=lambda t: t[1][0])
    print(f"Sorted turbines by x-coordinate: {sorted_turbines}")

    results = {}  # Store turbine results

    for j, (j_id, (x_j, y_j)) in enumerate(sorted_turbines):
        total_deficit_squared = 0  # Quadratic summation of wake deficits

        print(f"Processing turbine {j_id} at ({x_j}, {y_j})")

        for i, (i_id, (x_i, y_i)) in enumerate(sorted_turbines[:j]):  # Only upstream turbines
            x_ij = x_j - x_i  # Downstream distance
            d_ij = abs(y_j - y_i)  # Perpendicular distance

            if x_ij > 0:
                v_i = results.get(i_id, {}).get('v_j', v_0)  # Use previous turbine wind speed

                # Calculate wake effect
                v_j_xij, _, _, _, _, _, _ = multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij)
                wake_deficit = (v_i - v_j_xij) / v_0  # Normalized wake deficit
                total_deficit_squared += wake_deficit**2  # Quadratic summation

        # Compute effective wind speed and standard deviation
        total_deficit = np.sqrt(total_deficit_squared)
        v_j = v_0 * (1 - total_deficit)
        sigma_j = (v_j / v_0) * sigma_0  # Scale standard deviation

        # Compute Weibull shape factor (k) & scale factor (λ)
        k_j, lambda_j = calculate_weibull_params_vj(v_j, sigma_j)

        print(f"Turbine {j_id}: v_j = {v_j:.3f}, σ_j = {sigma_j:.3f}, k = {k_j:.3f}, λ = {lambda_j:.3f}")

        # Store turbine results
        results[j_id] = {'v_j': v_j, 'sigma_j': sigma_j, 'k': k_j, 'lambda': lambda_j}

    return results

import xlwings as xw

def mwm_identical_turbines_BEM(file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions):
    """
    Multiple Wake Model with Weibull Distribution Scaling and BEM Spreadsheet Integration.

    Parameters:
    - file_path (str): Path to the BEM spreadsheet.
    - sheet_name (str): Name of the sheet to interact with.
    - Ct (float): Thrust coefficient.
    - r_0 (float): Rotor radius (m).
    - z (float): Hub height (m).
    - z_0 (float): Surface roughness length (m).
    - turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions.

    Returns:
    - dict: Contains average energy yield (MW/hr), total annual energy yield (GWhr/year), and capacity factor for each turbine.
    """
    # Compute freestream wind speed and standard deviation at height z
    v_0 = calculate_mean_wind_speed(z)
    sigma_0 = calculate_std_dev(z)

    # Calculate Weibull parameters for each turbine using the wake model
    weibull_results = mwm_identical_turbines_WD(Ct, r_0, z, z_0, turbine_positions)

    # Open the BEM Excel workbook
    app = xw.App(visible=False)
    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]

    # Prepare results dictionary
    bem_results = {}

    for turbine_id, data in weibull_results.items():
        # Write Weibull coefficients (k and λ) to the spreadsheet
        sheet["G15"].value = data["k"]  # Shape factor
        sheet["G16"].value = data["lambda"]  # Scale factor

        # Force recalculation
        _ = sheet["G4"].value  # Access dependent cell to trigger recalculation

        # Read results from spreadsheet
        average_energy_yield = sheet["G4"].value  # MW/hr
        total_energy_yield = sheet["G7"].value  # GWhr/year
        capacity_factor = sheet["G10"].value  # Fraction

        # Store results in dictionary
        bem_results[turbine_id] = {
            "Average Energy Yield (MW/hr)": average_energy_yield,
            "Total Energy Yield (GWhr/year)": total_energy_yield,
            "Capacity Factor": capacity_factor
        }

    # Close the workbook and Excel app
    wb.close()
    app.quit()

    return bem_results

import numpy as np
from math import radians, cos, sin

def transform_turbine_positions(turbine_positions, theta):
    """
    Transform turbine positions based on the wind direction angle θ.

    Parameters:
    - turbine_positions (dict): Original turbine positions as {ID: (x, y)}.
    - theta (float): Wind direction angle in degrees (positive for clockwise rotation).

    Returns:
    - dict: Transformed turbine positions as {ID: (x', y')}.
    """
    theta_rad = -radians(theta)  # Convert angle to radians

    transformed_positions = {}
    for turbine_id, (x, y) in turbine_positions.items():
        # Apply rotation matrix
        x_prime = x * cos(theta_rad) + y * sin(theta_rad)
        y_prime = -x * sin(theta_rad) + y * cos(theta_rad)
        transformed_positions[turbine_id] = (x_prime, y_prime)

    return transformed_positions


def mwm_identical_turbines_BEM_any_direction(file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, theta):
    """
    Multiple Wake Model with Weibull Distribution Scaling and BEM Spreadsheet Integration.

    Parameters:
    - file_path (str): Path to the BEM spreadsheet.
    - sheet_name (str): Name of the sheet to interact with.
    - Ct (float): Thrust coefficient.
    - r_0 (float): Rotor radius (m).
    - z (float): Hub height (m).
    - z_0 (float): Surface roughness length (m).
    - turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions.

    Returns:
    - dict: Contains average energy yield (MW/hr), total annual energy yield (GWhr/year), and capacity factor for each turbine.
    """
    # Compute freestream wind speed and standard deviation at height z
    v_0 = calculate_mean_wind_speed(z)
    sigma_0 = calculate_std_dev(z)

    transformed_positions = transform_turbine_positions(turbine_positions, theta)

    # Calculate Weibull parameters for each turbine using the wake model for transformed positions
    weibull_results = mwm_identical_turbines_WD(Ct, r_0, z, z_0, transformed_positions)

    # Open the BEM Excel workbook
    app = xw.App(visible=False)
    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]

    # Prepare results dictionary
    bem_results = {}

    for turbine_id, data in weibull_results.items():
        # Write Weibull coefficients (k and λ) to the spreadsheet
        sheet["G15"].value = data["k"]  # Shape factor
        sheet["G16"].value = data["lambda"]  # Scale factor

        # Force recalculation
        _ = sheet["G4"].value  # Access dependent cell to trigger recalculation

        # Read results from spreadsheet
        average_energy_yield = sheet["G4"].value  # MW/hr
        total_energy_yield = sheet["G7"].value  # GWhr/year
        capacity_factor = sheet["G10"].value  # Fraction

        # Store results in dictionary
        bem_results[turbine_id] = {
            "Average Energy Yield (MW/hr)": average_energy_yield,
            "Total Energy Yield (GWhr/year)": total_energy_yield,
            "Capacity Factor": capacity_factor
        }

    # Close the workbook and Excel app
    wb.close()
    app.quit()

    return bem_results
