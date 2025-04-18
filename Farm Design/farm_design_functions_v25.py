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
        #print("Case: Complete shadowing")
        v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij) ** 2)
        return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i

    # Case 2: No shadowing (no overlap between wake and turbine rotor)
    elif d_ij >= (r_0 + r_i_xij):
        A_shadow_i = 0  # No shadowing occurs
        no_shadowing = True
        #("Case: No shadowing")
        return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i

    # Case 3: Partial shadowing (partial overlap)
    else:
        #print("Case: Partial shadowing")
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
                #print(f"A_shadow_i/A_0 (normalised shadowed area): {A_shadow_i/A_0}")
                v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij) ** 2 * (A_shadow_i / A_0))
                return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i
            except ValueError:
                A_shadow_i = 0  # Handle invalid geometry
        else:
            A_shadow_i = 0  # No valid overlap

    # Default return for no valid overlap or partial shadowing calculation errors
    no_shadowing = True
    #print("Case: No valid overlap (default case)")
    return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i


def mwm_identical_turbines(v_0, Ct, r_0, z, z_0, turbine_positions):
    """
    Multiple Wake Model for a Wind Farm of Identical Turbines (Using multiple_wake_model_ij).
    """
    # Sort turbines by upstream position (x-coordinate)
    sorted_turbines = sorted(turbine_positions.items(), key=lambda t: t[1][0])  # Sort by x-coordinate
    #print(f"Sorted turbines by x-coordinate: {sorted_turbines}")

    effective_wind_speeds = {}  # To store results

    for j, (j_id, (x_j, y_j)) in enumerate(sorted_turbines):
        total_deficit_squared = 0  # Use quadratic summation of wake deficits
        #print(f"Considering turbine {j_id} at position ({x_j}, {y_j})")

        for i, (i_id, (x_i, y_i)) in enumerate(sorted_turbines[:j]):  # Only consider upstream turbines
            x_ij = x_j - x_i  # Downstream distance
            #print(f"x_ij (downstream distance): {x_ij}")
            d_ij = abs(y_j - y_i)  # Perpendicular distance
            #print(f"d_ij (perpendicular distance): {d_ij}")

            if x_ij > 0:  # Only consider turbines upstream
                # Use the effective wind speed at turbine i
                v_i = effective_wind_speeds.get(i_id, v_0)

                #print(f"Considering turbine pair; {i_id} -> {j_id}")

                # Calculate wake effect using multiple_wake_model_ij
                v_j_xij, _, _, _, _, _, _ = multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij)
                #print(f"v_j_xij (wind speed at j due to i): {v_j_xij}")

                # Calculate wake deficit for turbine j from turbine i
                wake_deficit = (v_i - v_j_xij) / v_0  # Normalized wake deficit
               # print(f"wake_deficit: {wake_deficit}")
                total_deficit_squared += wake_deficit**2  # Quadratic summation

        # Combine wake deficits using energy superposition
        total_deficit = np.sqrt(total_deficit_squared)
        effective_wind_speeds[j_id] = v_0 * (1 - total_deficit)
        #print(f"Turbine {j_id} effective wind speed: {effective_wind_speeds[j_id]}")
       # print(" ")

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
    #print(f"Sorted turbines by x-coordinate: {sorted_turbines}")

    results = {}  # To store effective wind speeds and standard deviations

    for j, (j_id, (x_j, y_j)) in enumerate(sorted_turbines):
        total_deficit_squared = 0  # Use quadratic summation of wake deficits
       # print(f"Considering turbine {j_id} at position ({x_j}, {y_j})")

        for i, (i_id, (x_i, y_i)) in enumerate(sorted_turbines[:j]):  # Only consider upstream turbines
            x_ij = x_j - x_i  # Downstream distance
            d_ij = abs(y_j - y_i)  # Perpendicular distance

            if x_ij > 0:  # Only consider turbines upstream
                # Use the effective wind speed at turbine i
                v_i = results.get(i_id, {}).get('v_j', v_0)

               # print(f"Considering turbine pair; {i_id} -> {j_id}")

                # Calculate wake effect using multiple_wake_model_ij
                v_j_xij, _, _, _, _, _, _ = multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij)
               # print(f"v_j_xij (wind speed at j due to i): {v_j_xij}")

                # Calculate wake deficit for turbine j from turbine i
                wake_deficit = (v_i - v_j_xij) / v_0  # Normalized wake deficit
               # print(f"wake_deficit: {wake_deficit}")
                total_deficit_squared += wake_deficit**2  # Quadratic summation

        # Combine wake deficits using energy superposition
        total_deficit = np.sqrt(total_deficit_squared)
        v_j = v_0 * (1 - total_deficit)  # Effective mean wind speed

        # Scale the standard deviation based on the mean wind speed
        sigma_j = (v_j / v_0) * sigma_0  # Standard deviation scaling
        #print(f"Turbine {j_id} effective wind speed: {v_j}, standard deviation: {sigma_j}")
        #print(" ")

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
    #print(f"Sorted turbines by x-coordinate: {sorted_turbines}")

    results = {}  # Store turbine results

    for j, (j_id, (x_j, y_j)) in enumerate(sorted_turbines):
        total_deficit_squared = 0  # Quadratic summation of wake deficits

        #print(f"Processing turbine {j_id} at ({x_j}, {y_j})")

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

        #print(f"Turbine {j_id}: v_j = {v_j:.3f}, σ_j = {sigma_j:.3f}, k = {k_j:.3f}, λ = {lambda_j:.3f}")

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
    #app = xw.App(visible=False)
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
    #wb.close()
    #app.quit()

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
    theta_rad = radians(theta)  # Convert angle to radians

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
    #app = xw.App(visible=False)
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
    #wb.close()
    #app.quit()

    return bem_results

def energy_yield_with_wind_resource(file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose, primary_direction=240):
    """
    Compute annual energy yield for each turbine considering wind direction frequency.

    Parameters:
    - file_path (str): Path to the BEM spreadsheet.
    - sheet_name (str): Name of the sheet to interact with.
    - Ct (float): Thrust coefficient.
    - r_0 (float): Rotor radius (m).
    - z (float): Hub height (m).
    - z_0 (float): Surface roughness length (m).
    - turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions.
    - wind_rose (dict): Dictionary where keys are wind directions (degrees), and values are frequency percentages (0-1).
    - primary_direction (int): The primary wind direction in degrees.

    Returns:
    - dict: Annual energy yield (GWhr/year) for each turbine.
    """
    # Initialize annual energy yield dictionary for each turbine
    annual_yield = {turbine_id: 0 for turbine_id in turbine_positions.keys()}

    # Iterate through each wind direction and its frequency in the wind rose
    for direction, frequency in wind_rose.items():
        #print(f"Processing wind direction: {direction}°")
        #print(f"Wind direction frequency: {frequency}")
        # Calculate theta as deviation from primary direction
        theta = -(direction - primary_direction) % 360  # Ensure theta is within [0, 360)
        if theta > 180:  # Convert angles >180° to negative for symmetry (e.g., 270° -> -90°)
            theta += 360

        # Calculate energy yield for the current wind direction
        direction_yield = mwm_identical_turbines_BEM_any_direction(
            file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, theta
        )

        # Update the annual yield by weighting with wind direction frequency
        for turbine_id, data in direction_yield.items():
            annual_yield[turbine_id] += data["Total Energy Yield (GWhr/year)"] * frequency
            #print(f"Turbine {turbine_id}: {data['Total Energy Yield (GWhr/year)']} GWhr/year")

    return annual_yield

import pandas as pd
from pyproj import Proj, Transformer

def get_turbine_long_lat(csv_file):
    """
    Convert UTM coordinates of turbines to latitude and longitude.

    Parameters:
    - csv_file (str): Path to CSV file containing UTM Easting/Northing values.

    Returns:
    - dict: Dictionary with turbine ID as key and (longitude, latitude) tuple as value.
    """
    
    # Load turbine positions from CSV
    df = pd.read_csv(csv_file, header=None, names=["Easting", "Northing"])

    # Define the projection (Sheringham Shoal is in UTM Zone 31N, EPSG:32631)
    utm_proj = Proj(proj="utm", zone=31, datum="WGS84")
    wgs84_proj = Proj(proj="latlong", datum="WGS84")

    # Transformer for coordinate conversion
    transformer = Transformer.from_proj(utm_proj, wgs84_proj)

    # Dictionary to store turbine coordinates
    turbine_coordinates = {}

    for index, row in df.iterrows():
        turbine_id = index + 1  # Assigning IDs starting from 1
        easting = row["Easting"]
        northing = row["Northing"]

        # Convert UTM to Longitude/Latitude
        longitude, latitude = transformer.transform(easting, northing)

        # Store in dictionary
        turbine_coordinates[turbine_id] = (longitude, latitude)

    return turbine_coordinates



import numpy as np
import random
from scipy.spatial import distance
from shapely.geometry import Polygon, Point

def generate_random_layout(boundary, min_spacing, num_turbines_range):
    """
    Generate a valid random turbine layout with a random number of turbines.

    Parameters:
    - boundary (numpy array): UTM boundary coordinates (Easting, Northing).
    - min_spacing (float): Minimum allowed spacing between turbines (meters).
    - num_turbines_range (tuple): Min and max number of turbines allowed.

    Returns:
    - dict: Turbine positions as {ID: (x, y)}.
    """
    num_turbines = np.random.randint(num_turbines_range[0], num_turbines_range[1] + 1)  # Random number of turbines
    print(f"Number of turbines to randombly generate: {num_turbines}")
    turbines = {}
    
    # Create a polygon from the boundary points
    boundary_polygon = Polygon(boundary)

    id_counter = 1  # Start turbine IDs from 1

    while len(turbines) < num_turbines:
        # Generate a random point within boundary limits
        x_rand = np.random.uniform(min(boundary[:, 0]), max(boundary[:, 0]))
        y_rand = np.random.uniform(min(boundary[:, 1]), max(boundary[:, 1]))

        # Ensure the point is inside the boundary polygon
        if boundary_polygon.contains(Point(x_rand, y_rand)):
            # Check if spacing constraint is met
            if all(distance.euclidean((x_rand, y_rand), t) >= min_spacing for t in turbines.values()):
                turbines[id_counter] = (x_rand, y_rand)
                id_counter += 1  # Increment ID
                #print(f"Added turbine at ({x_rand}, {y_rand}) for ID: {id_counter}")

    return turbines

def initialise_population(population_size, boundary, min_spacing, num_turbines_range):
    """
    Generate an initial population of turbine layouts with a variable number of turbines.

    Parameters:
    - population_size (int): Number of random layouts to generate.
    - boundary (numpy array): UTM boundary coordinates.
    - min_spacing (float): Minimum spacing between turbines.
    - num_turbines_range (tuple): Min and max turbine count.

    Returns:
    - list: A list of dictionaries, each representing a turbine layout.
    """
    return [generate_random_layout(boundary, min_spacing, num_turbines_range) for _ in range(population_size)]

import numpy as np
from scipy.spatial import distance_matrix

def calculate_average_turbine_spacing(layout_dict, k=4):
    """
    Calculates the average distance to the k nearest neighbours for each turbine.

    Args:
        layout_dict (dict): Turbine ID to (x, y) coordinates in metres.
        k (int): Number of nearest neighbours to consider (default is 4).

    Returns:
        float: Average spacing to k nearest neighbours (m)
        float: Minimum average spacing across turbines (m)
        float: Maximum average spacing across turbines (m)
        int: Number of turbines
    """
    coords = np.array(list(layout_dict.values()))
    n = coords.shape[0]

    if k >= n:
        raise ValueError("k must be less than the number of turbines.")

    # Compute full distance matrix
    dist_mat = distance_matrix(coords, coords)
    np.fill_diagonal(dist_mat, np.inf)  # ignore self-distances

    # Get distances to k nearest neighbours per turbine
    nearest_k_distances = np.sort(dist_mat, axis=1)[:, :k]
    avg_per_turbine = np.mean(nearest_k_distances, axis=1)

    # Compute global stats
    avg_spacing = float(np.mean(avg_per_turbine))
    min_spacing = float(np.min(avg_per_turbine))
    max_spacing = float(np.max(avg_per_turbine))

    return avg_spacing, min_spacing, max_spacing, n

import math

def LCOE(file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose, primary_direction=240):
    """
    UPDATED LCOE FUNCTION, including cabling costs and more accurate component breakdown.
    Compute the lifetime Levelized Cost of Energy (LCOE) for a given wind turbine arrangement.

    Parameters:
    - file_path (str): Path to the BEM spreadsheet.
    - sheet_name (str): Name of the sheet to interact with.
    - Ct (float): Thrust coefficient.
    - r_0 (float): Rotor radius (m).
    - z (float): Hub height (m).
    - z_0 (float): Surface roughness length (m).
    - turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions.
    - wind_rose (dict): Wind direction frequencies.
    - primary_direction (int): Primary wind direction.

    Returns:
    - float: The computed LCOE (£/MWhr).
    """


    # Constants
    CAPEX_per_turbine_excluding_PT = 91_481_773 
    OPEX_per_turbine_per_year = 2_613_208

    TURBINE_CAPACITY_MW = 25.157  #
    discount_rate = 0.07  # 7% discount rate
    lifetime = 40 

    C_export = 86_400_000  # Cost of export cable (£)
    array_cable_per_km = 630_000  # Cost of array cable per km (£)

    num_turbines = len(turbine_positions)  # Number of turbines in the layout
    print(f"Number of turbines in current individual being evaluated: {num_turbines}")

    # Compute total energy yield for this layout
    total_energy_yield = sum(energy_yield_with_wind_resource(
        file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose, primary_direction
    ).values())  # Sum all turbine yields

    total_energy_yield = total_energy_yield * 1000  # Convert GWhr to MWhr
    print(f"Total energy yield for current individual: {total_energy_yield:.4f} MWhr")


    # ***Calculating cabling costs, C_PT***

    # Calculate the total length of array cables
    average_turbine_spacing,_,_,_ = calculate_average_turbine_spacing(turbine_positions) # Average spacing between turbines in metres

    average_turbine_spacing = average_turbine_spacing / 1000  # Convert to km

    L_string = 8 * average_turbine_spacing  # Length of string (8 turbines per string)
    N_string = math.ceil(len(turbine_positions) / 8)  # Number of strings = total turbines / 8

    L_array = N_string * L_string * 1.15 # Total length of array cables with 15% contingency

    # Calculate the cost of array cables
    C_array = L_array * array_cable_per_km
    #print(f"Cost of array cables in millions: {C_array/1_000_000:.4f} £")

    # Calculate offshore substations costs
    cost_of_1_offshore_substation = 583_000 + 0.1079 * len(turbine_positions)  * TURBINE_CAPACITY_MW
    C_off_subs = 2 * cost_of_1_offshore_substation

    # Calculate onshore substation cost
    C_on_subs = cost_of_1_offshore_substation/2

    #C_PT = C_array + C_export + C_off_subs + C_on_subs
    C_PT = C_array + C_export + C_off_subs + C_on_subs + 200_000_000 # Add 100M for contingency

    # Investment costs
    I_1 = CAPEX_per_turbine_excluding_PT * num_turbines + C_PT

    # Annual recurring costs
    A_k = OPEX_per_turbine_per_year * num_turbines

    # Initialising LCOE numerator and denominator
    LCOE_numerator = 0
    LCOE_denominator = 0

    # Calculate LCOE using sum (for loop)
    for k in range(1, lifetime+1):
        if k == 1:
            I_k = I_1
        else:
            I_k = 0
        
        LCOE_numerator += (I_k + A_k)/((1 + discount_rate)**k) 

        #LCOE_denominator += (total_energy_yield)/((1 + discount_rate)**k)
        LCOE_denominator += (total_energy_yield)*1.33
        #print(f"Year {k}: LCOE Numerator: {LCOE_numerator}, LCOE Denominator: {LCOE_denominator}")

    LCOE_value = LCOE_numerator / LCOE_denominator
    print(f"LCOE for current individual with {num_turbines} turbines: {LCOE_value:.5f} £/MWhr")
    print("--------------------------------------------")
    print(" ")

    return LCOE_value

def reproduction_selection(population, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose):
    """
    Select the top 2 parents from the population based on their LCOE fitness.

    Parameters:
    - population (list): A list of turbine layout dictionaries.
    - file_path (str): Path to the BEM spreadsheet.
    - sheet_name (str): Name of the sheet to interact with.
    - Ct (float): Thrust coefficient.
    - r_0 (float): Rotor radius (m).
    - z (float): Hub height (m).
    - z_0 (float): Surface roughness length (m).
    - wind_rose (dict): Wind direction frequencies.

    Returns:
    - list: Two best parents selected based on LCOE.
    """

    # Compute LCOE for all layouts in the population
    fitness_scores = {tuple(layout.items()): LCOE(file_path, sheet_name, Ct, r_0, z, z_0, layout, wind_rose) for layout in population}
    #print(f"Fitness Scores: {fitness_scores}")

    # Select top 2 layouts with lowest LCOE (best fitness)
    top_2 = sorted(fitness_scores, key=fitness_scores.get)[:2]

    # Convert back to dictionary format
    return [dict(layout) for layout in top_2]

import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import distance

import numpy as np
from shapely.geometry import Polygon, Point


import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import distance
import random

def random_crossover(parent_1, parent_2, boundary, min_spacing):
    """
    Perform crossover between two wind farm layouts by selecting a valid region 
    that covers 50% to 75% of the total Sheringham Shoal site.

    Parameters:
    - parent_1 (dict): Fitter parent's turbine positions {ID: (x, y)}.
    - parent_2 (dict): Less fit parent's turbine positions {ID: (x, y)}.
    - boundary (numpy array): UTM boundary coordinates.
    - min_spacing (float): Minimum spacing between turbines.

    Returns:
    - dict: Child layout (new turbine positions with renumbered IDs).
    - Polygon: Selected crossover region for visualization.
    """

    # Convert boundary to a polygon
    boundary_polygon = Polygon(boundary)
    total_site_area = boundary_polygon.area  # Compute total Sheringham Shoal site area

    # Initialize valid crossover region
    valid_region = None
    crossover_proportion = 0

    # Try multiple times to ensure a valid proportion
    for _ in range(110):  # Allow multiple attempts to find a good region
        # Define random region as a proportion of total area (50% to 75%)
        proportion = np.random.uniform(0.5, 0.75)
        

        # Generate a bounding box within the site
        min_x, min_y, max_x, max_y = boundary[:, 0].min(), boundary[:, 1].min(), boundary[:, 0].max(), boundary[:, 1].max()
        region_x_min = np.random.uniform(min_x, min_x + (max_x - min_x) * (1 - proportion))
        region_x_max = np.random.uniform(region_x_min + (max_x - region_x_min) * 0.5, max_x)
        region_y_min = np.random.uniform(min_y, min_y + (max_y - min_y) * (1 - proportion))
        region_y_max = np.random.uniform(region_y_min + (max_y - region_y_min) * 0.5, max_y)

        crossover_region = Polygon([(region_x_min, region_y_min), (region_x_max, region_y_min),
                                    (region_x_max, region_y_max), (region_x_min, region_y_max)])

        # Ensure it remains inside the boundary
        if boundary_polygon.contains(crossover_region):
            valid_region = crossover_region
        else:
            valid_region = boundary_polygon.intersection(crossover_region)

        # Compute actual area proportion
        crossover_area = valid_region.area
        crossover_proportion = crossover_area / total_site_area

        # Accept only if it falls within the 50% to 75% range
        if 0.5 <= crossover_proportion <= 0.75:
            break  # Found a valid region, exit loop

    # Debugging output: Print the area ratio
    #print(f"Crossover Region Area: {crossover_area:.2f} m²")
    #print(f"Total Sheringham Shoal Site Area: {total_site_area:.2f} m²")
    #print(f"Proportion of Site Used for Crossover: {crossover_proportion:.2%}")

    # Initialize child layout
    child_layout = {}
    used_positions = set()

    # Inherit turbines inside the crossover region from the fitter parent
    for tid, (x, y) in parent_1.items():
        if valid_region.contains(Point(x, y)):
            child_layout[tid] = (x, y)
            used_positions.add((x, y))  # Track positions to avoid duplication

    # Fill remaining turbines from the less fit parent (outside crossover region)
    for tid, (x, y) in parent_2.items():
        if (x, y) not in used_positions and not valid_region.contains(Point(x, y)):
            # Ensure spacing is maintained
            if all(distance.euclidean((x, y), t) >= min_spacing for t in child_layout.values()):
                new_id = max(child_layout.keys(), default=0) + 1  # Assign new unique ID
                child_layout[new_id] = (x, y)
                used_positions.add((x, y))

    return child_layout, valid_region

def next_generation(population, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose, boundary, min_spacing):
    """
    Generate the next generation of wind farm layouts using weighted reproduction.

    Parameters:
    - population (list): List of turbine layout dictionaries.
    - file_path (str): Path to the BEM spreadsheet.
    - sheet_name (str): Name of the sheet to interact with.
    - Ct (float): Thrust coefficient.
    - r_0 (float): Rotor radius (m).
    - z (float): Hub height (m).
    - z_0 (float): Surface roughness length (m).
    - wind_rose (dict): Wind direction frequencies.
    - boundary (numpy array): UTM boundary coordinates.
    - min_spacing (float): Minimum spacing between turbines.

    Returns:
    - list: New population of 10 wind farm layouts.
    """

    # **Step 1: Compute fitness scores (LCOE)**
    fitness_scores = {tuple(layout.items()): LCOE(file_path, sheet_name, Ct, r_0, z, z_0, layout, wind_rose) for layout in population}
    
    # **Step 2: Select top 3 fittest layouts**
    sorted_population = sorted(fitness_scores, key=fitness_scores.get)
    parent_1 = dict(sorted_population[0])  # Most fit
    parent_2 = dict(sorted_population[1])  # Second most fit
    parent_3 = dict(sorted_population[2])  # Third most fit

    # **Step 3: Generate 10 new layouts**
    new_population = [
        parent_1,  # Keep most fit
        parent_2,  # Keep second most fit
    ]

    # **Step 4: Reproduce using weighted crossover**
    for _ in range(3):  
        child, _ = random_crossover(parent_1, parent_2, boundary, min_spacing)
        new_population.append(child)
    
    for _ in range(3):  
        child, _ = random_crossover(parent_1, parent_3, boundary, min_spacing)
        new_population.append(child)

    for _ in range(2):  
        child, _ = random_crossover(parent_2, parent_3, boundary, min_spacing)
        new_population.append(child)

    return new_population

import numpy as np
import random
from shapely.geometry import Polygon, Point
from scipy.spatial import distance

def mutate_layout(turbine_layout, boundary, min_spacing, r_0):
    """
    Apply mutation to a wind farm layout. Mutations include turbine position shift, addition, and removal.

    Parameters:
    - turbine_layout (dict): Dictionary of turbine positions {ID: (x, y)}.
    - boundary (numpy array): UTM boundary coordinates.
    - min_spacing (float): Minimum allowed spacing between turbines.
    - r_0 (float): Rotor radius (m).

    Returns:
    - dict: Mutated turbine layout.
    """
    boundary_polygon = Polygon(boundary)
    mutated_layout = turbine_layout.copy()
    turbine_ids = list(mutated_layout.keys())

    # **Turbine Position Shift (15% chance)**
    if np.random.rand() < 0.15:
        num_shift = np.random.randint(1, 11)  # Select 1-10 turbines to shift
        #print(f"Shifting {num_shift} turbines")
        selected_turbines = random.sample(turbine_ids, min(num_shift, len(turbine_ids)))

        for tid in selected_turbines:
            x, y = mutated_layout[tid]
            shift_x = np.random.uniform(-3 * r_0, 3 * r_0)
            shift_y = np.random.uniform(-3 * r_0, 3 * r_0)

            new_x, new_y = x + shift_x, y + shift_y
            if boundary_polygon.contains(Point(new_x, new_y)) and all(
                distance.euclidean((new_x, new_y), pos) >= min_spacing for pos in mutated_layout.values() if pos != (x, y)
            ):
                mutated_layout[tid] = (new_x, new_y)

    # **Turbine Addition (10% chance)**
    if np.random.rand() < 0.10:
        num_add = np.random.randint(1, 6)  # Select 1-5 turbines to add
        #print(f"Adding {num_add} turbines")
        for _ in range(num_add):
            while True:
                new_x = np.random.uniform(boundary[:, 0].min(), boundary[:, 0].max())
                new_y = np.random.uniform(boundary[:, 1].min(), boundary[:, 1].max())

                if boundary_polygon.contains(Point(new_x, new_y)) and all(
                    distance.euclidean((new_x, new_y), pos) >= min_spacing for pos in mutated_layout.values()
                ):
                    new_id = max(mutated_layout.keys(), default=0) + 1
                    mutated_layout[new_id] = (new_x, new_y)
                    break

    # **Turbine Removal (10% chance)**
    if np.random.rand() < 0.10 and len(mutated_layout) > 1:
        num_remove = np.random.randint(1, 6)  # Select 1-5 turbines to remove
        remove_turbines = random.sample(turbine_ids, min(num_remove, len(turbine_ids)))
       # print(f"Removing {num_remove} turbines")
        for tid in remove_turbines:
            if tid in mutated_layout:
                del mutated_layout[tid]

    return mutated_layout

def next_generation_with_mutation(population, fitness_scores, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose, boundary, min_spacing):
    """
    Generate the next generation of wind farm layouts, including mutation.
    
    Now uses precomputed fitness scores to avoid redundant LCOE calculations.
    """
    # Sort population by fitness (ascending LCOE is better)
    sorted_population = sorted(fitness_scores, key=fitness_scores.get)

    # Select top 3 fittest parents
    parent_1 = dict(sorted_population[0])  # Best
    parent_2 = dict(sorted_population[1])  # Second best
    parent_3 = dict(sorted_population[2])  # Third best

    # Generate new population
    new_population = [parent_1, parent_2]  # Keep top 2 parents

    # Generate children through crossover
    for _ in range(3):
        child, _ = random_crossover(parent_1, parent_2, boundary, min_spacing)
        new_population.append(child)

    for _ in range(3):
        child, _ = random_crossover(parent_1, parent_3, boundary, min_spacing)
        new_population.append(child)

    for _ in range(2):
        child, _ = random_crossover(parent_2, parent_3, boundary, min_spacing)
        new_population.append(child)

    # Apply mutation to all children (excluding parents)
    for i in range(2, len(new_population)):
        new_population[i] = mutate_layout(new_population[i], boundary, min_spacing, r_0)

    return new_population

def genetic_algorithm(file_path, sheet_name, boundary, min_spacing, num_turbines_range, 
                      Ct, r_0, z, z_0, wind_rose, max_generations=50, convergence_threshold=0.001):
    """
    Run the Genetic Algorithm to optimize wind farm layouts.

    **Uses a persistent Excel session for efficiency.**
    """

    # **Step 1: Open Excel Session Once**
    app = xw.App(visible=False, add_book=False)
    app.display_alerts = False  
    app.screen_updating = False  
    wb = xw.Book(file_path)
    sheet = wb.sheets[sheet_name]

    try:
        # **Step 2: Initialize Population**
        population_size = 10
        population = initialise_population(population_size, boundary, min_spacing, num_turbines_range)

        # Track best layouts and LCOEs
        best_layouts = []
        best_LCOEs = [] # Track best LCOE per generation
        second_best_LCOEs = [] # Track second best LCOE per generation
        recent_LCOEs = []  # Track last 5 generations for convergence

        # **Step 3: Evaluate Fitness for Initial Population**
        print("Evaluating initial population fitness (LCOE)...")
        fitness_scores = {}

        for layout in population:
            layout_tuple = tuple(layout.items())  # Convert dict to hashable format
            if layout_tuple in fitness_scores:
                print(f"Using cached LCOE: {fitness_scores[layout_tuple]:.4f} for layout.")
                print("")
            else:
                lcoe_value = LCOE(file_path, sheet_name, Ct, r_0, z, z_0, layout, wind_rose)
                fitness_scores[layout_tuple] = lcoe_value  # Store in dict

        # **Step 4: Run Genetic Algorithm**
        lcoe_cache = {}  # Store previously computed LCOE values

        for generation in range(1, max_generations + 1):
            print(f"\n======================== Generation {generation} ========================")

            # **Step 5: Generate Next Generation with Mutation**
            new_population = next_generation_with_mutation(population, fitness_scores, file_path, sheet_name, Ct, r_0, z, z_0, wind_rose, boundary, min_spacing)

            print(f"Max turbines in new gen: {max(len(layout) for layout in new_population)}, Min turbines: {min(len(layout) for layout in new_population)}")

            # **Step 6: Evaluate Fitness of New Population (with caching)**
            print("Evaluating fitness (LCOE) for new population...")
            print("")
            fitness_scores = {}

            for layout in new_population:
                layout_tuple = tuple(layout.items())  # Convert dict to hashable format
                print(f"Evaluating layout {new_population.index(layout) + 1} in new population of {len(new_population)}")
                #print("")
                if layout_tuple in lcoe_cache:
                    fitness_scores[layout_tuple] = lcoe_cache[layout_tuple]
                    print(f"Turbine layout already evaluated. Using cached LCOE: {fitness_scores[layout_tuple]:.2f}")
                else:
                    lcoe_value = LCOE(file_path, sheet_name, Ct, r_0, z, z_0, layout, wind_rose)
                    fitness_scores[layout_tuple] = lcoe_value
                    lcoe_cache[layout_tuple] = lcoe_value  # Store in cache

            # **Step 7: Select the Fittest Individual**
            best_layout_tuple = min(fitness_scores, key=fitness_scores.get)
            best_LCOE = fitness_scores[best_layout_tuple]

                        # Sort the dictionary by values and get the keys in ascending order
            sorted_layouts = sorted(fitness_scores, key=fitness_scores.get)

            # Get the second smallest layout and its corresponding LCOE
            second_best_layout_tuple = sorted_layouts[1]  # Index 1 for the second smallest
            second_best_LCOE = fitness_scores[second_best_layout_tuple]

            

            # Store best layout and LCOE
            best_layouts.append((dict(best_layout_tuple), best_LCOE, generation))
            best_LCOEs.append(best_LCOE)

            # Store second best LCOE
            second_best_LCOEs.append(second_best_LCOE)

            print(f"Best LCOE this generation: {best_LCOE:.2f} £/MWhr, with number of turbines: {len(best_layout_tuple)}")
            print(f"Best Layout: {dict(best_layout_tuple)}")

            # **Step 8: Convergence Check**
            recent_LCOEs.append(best_LCOE)
            if len(recent_LCOEs) > 5:
                recent_LCOEs.pop(0)  # Keep only the last 5 LCOEs

            if len(recent_LCOEs) == 5:
                LCOE_change = abs(recent_LCOEs[-1] - recent_LCOEs[0]) / recent_LCOEs[0]
                if LCOE_change < convergence_threshold:
                    print(f"✅ Convergence reached: LCOE change < {convergence_threshold * 100:.2f}% over last 5 generations.")
                    break

            # Move to the next generation
            population = new_population

        # **Step 9: Return Results**
        best_layouts_sorted = sorted(best_layouts, key=lambda x: x[1])[:5]  # Keep top 5
        final_best_layout = dict(best_layouts_sorted[0][0])
        final_best_LCOE = best_layouts_sorted[0][1]

        lowest_LCOE_per_gen = best_LCOEs
        second_lowest_LCOE_per_gen = second_best_LCOEs

        print(f"\n🎯 Final Best Layout Found in Generation {best_layouts_sorted[0][2]} with LCOE: {final_best_LCOE:.2f} £/MWhr")
    
        return best_layouts_sorted, final_best_layout, final_best_LCOE, lowest_LCOE_per_gen, second_lowest_LCOE_per_gen

    finally:
        # **Ensure Excel closes after the full algorithm completes**
        wb.close()
        app.quit()

import numpy as np

def define_parameter_space(rotor_diameter, A, B, alpha_deg, 
                           a_steps=10, b_steps=10, theta_steps=5):
    """
    Define the parameter space for regular wind farm layout.

    Parameters:
    - rotor_diameter (float): Rotor diameter (m).
    - A (float): Side length of boundary along row direction (m).
    - B (float): Side length of boundary along column direction (m).
    - alpha_deg (float): Boundary angle in degrees.
    - a_steps (int): Number of samples for row spacing a.
    - b_steps (int): Number of samples for column spacing b.
    - theta_steps (int): Number of orientation angles to test.

    Returns:
    - param_space (list): List of (a, b, theta_deg) tuples.
    """
    theta_tolerance = 15
    # Define limits
    D = rotor_diameter
    a_min = 3 * D
    a_max = 0.99 * A

    b_min = 3 * D
    b_max = 0.99 * B

    # Create uniform sample grids
    a_vals = np.linspace(a_min, a_max, a_steps)
    b_vals = np.linspace(b_min, b_max, b_steps)
    theta_vals = np.linspace(alpha_deg, alpha_deg - theta_tolerance, theta_steps)

    # Build full parameter space as combinations of (a, b, theta)
    param_space = [(a, b, theta) for a in a_vals for b in b_vals for theta in theta_vals]

    return param_space

import numpy as np

import numpy as np

import numpy as np

import numpy as np

import numpy as np

from shapely.geometry import Point, Polygon

def generate_regular_arrangement(a, b, theta_deg, A, B, alpha_deg):
    """
    Generate a regular parallelogram turbine layout grid anchored at a given point.

    Parameters:
    - a (float): Row spacing (m)
    - b (float): Column spacing (m)
    - theta_deg (float): Orientation angle in degrees.
    - A (float): Side length of boundary along row direction (m).
    - B (float): Side length of boundary along column direction (m).
    - alpha_deg (float): Boundary angle in degrees.
    
    Returns:
    - layout_dict (dict): {turbine_id: (x, y)} for turbines in the full parallelogram grid
    """
    theta_rad = np.radians(theta_deg)
    alpha_rad = np.radians(alpha_deg)

    phi_rad = (alpha_rad - theta_rad)/2 # Angle between the row direction and the boundary

    # Sheringham Shoal UTM boundary coordinates
    boundary_utm = np.array([
        (371526.684, 5893424.206),  # P1
        (378450.513, 5890464.698),  # P2
        (380624.287, 5884484.707),  # P3
        (373690.373, 5887454.043),  # P4
        (371526.684, 5893424.206)   # P1 again to close loop
    ])


    # Label key vertices
    P1 = boundary_utm[0]
    P2 = boundary_utm[1]
    P3 = boundary_utm[2]
    P4 = boundary_utm[3]  # use P4 not P3, since P4 shares vertex with P1
        
    # Define orign
    origin = P4  # Use P4 as origin

    # Define the vectors u and v
    u = np.array([P3[0] - P4[0], P3[1] - P4[1]]) # Vector from P4 to P3
    v = np.array([P1[0] - P4[0], P1[1] - P4[1]]) # Vector from P4 to P1

    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)

    # Define rotation matrix
    R_phi = np.array([[np.cos(phi_rad), -np.sin(phi_rad)], 
                      [np.sin(phi_rad), np.cos(phi_rad)]]) 
    
    R_minus_phi = np.array([[np.cos(-phi_rad), -np.sin(-phi_rad)], 
                      [np.sin(-phi_rad), np.cos(-phi_rad)]])
    
    # Calculate a_norm and b_norm
    a_norm = np.dot(R_phi, u_norm)
    b_norm = np.dot(R_minus_phi, v_norm)

    # Calculate a and b
    a_vector = a*a_norm
    b_vector = b*b_norm

    #Calculate number of rows and columns
    R = math.ceil(int(A/a))
    C = math.ceil(int(B/b))
    print(f"Number of rows: {R}, Number of columns: {C}")

    # Generate the layout
    layout_dict = {}
    turbine_id = 1

    for i in range(R+2):
        for j in range(C+2):
            x = origin[0] + i*a_vector[0] + j*b_vector[0]
            y = origin[1] + i*a_vector[1] + j*b_vector[1]
            layout_dict[turbine_id] = (x, y)
            turbine_id += 1
    
    # Truncate layout to fit within boundary
    boundary_poly = Polygon(boundary_utm)

    # Define safe buffered boundary
    safe_boundary = Polygon(boundary_utm).buffer(0.01)  # 1 cm buffer

    # Filter layout
    layout_dict_filtered = {
        tid: coords for tid, coords in layout_dict.items()
        if safe_boundary.contains(Point(coords))
    }

    print(f"Number of turbines in regular layout: {len(layout_dict_filtered)}")
    print(f"filtered layout: {layout_dict_filtered}")

    return layout_dict_filtered