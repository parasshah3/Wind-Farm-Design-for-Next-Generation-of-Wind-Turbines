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
        print(f"Processing wind direction: {direction}°")
        print(f"Wind direction frequency: {frequency}")
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

def LCOE(file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose, primary_direction=240):
    """
    Compute the annual Levelized Cost of Energy (LCOE) for a given wind turbine arrangement.

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
    CAPEX_PER_TURBINE = 3_000_000  # £3M per turbine
    TURBINE_COST_PER_MW = 1_500_000  # £1.5M per MW per turbine
    OPEX_PER_TURBINE = 400_000  # £400k per turbine per year
    TURBINE_CAPACITY_MW = 25.074  

    num_turbines = len(turbine_positions)  # Number of turbines in the layout

    # Compute total energy yield for this layout
    total_energy_yield = sum(energy_yield_with_wind_resource(
        file_path, sheet_name, Ct, r_0, z, z_0, turbine_positions, wind_rose, primary_direction
    ).values())  # Sum all turbine yields

    total_energy_yield = total_energy_yield * 1000  # Convert GWhr to MWhr

    # Compute total annual cost
    CAPEX_annual = (CAPEX_PER_TURBINE + TURBINE_COST_PER_MW * TURBINE_CAPACITY_MW) * num_turbines
    OPEX_annual = OPEX_PER_TURBINE * num_turbines
    total_annual_cost = CAPEX_annual + OPEX_annual

    # Compute LCOE (fitness function)
    LCOE_value = total_annual_cost / total_energy_yield if total_energy_yield > 0 else float('inf')

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
    print(f"Fitness Scores: {fitness_scores}")

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
import random
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull

import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import distance
import random

import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import distance
import random

import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import distance
import random

import numpy as np
from shapely.geometry import Polygon, Point
from scipy.spatial import distance
import random

def random_crossover(parent_1, parent_2, boundary, min_spacing):
    """
    Perform crossover between two wind farm layouts by selecting a random region
    and inheriting turbines while maintaining the spacing constraint.

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

    # Define random region as a proportion of total area (50% to 75%)
    proportion = np.random.uniform(0.5, 0.75)
    print(f"Proportion of Site Used for Crossover: {proportion:.2%}")

    # Generate a bounding polygon within the site
    min_x, min_y, max_x, max_y = boundary[:, 0].min(), boundary[:, 1].min(), boundary[:, 0].max(), boundary[:, 1].max()
    region_x_min = np.random.uniform(min_x, min_x + (max_x - min_x) * (1 - proportion))
    region_x_max = np.random.uniform(region_x_min + (max_x - region_x_min) * 0.5, max_x)
    region_y_min = np.random.uniform(min_y, min_y + (max_y - min_y) * (1 - proportion))
    region_y_max = np.random.uniform(region_y_min + (max_y - region_y_min) * 0.5, max_y)

    crossover_region = Polygon([(region_x_min, region_y_min), (region_x_max, region_y_min),
                                (region_x_max, region_y_max), (region_x_min, region_y_max)])

    # Ensure the region remains within the main site boundary
    if not boundary_polygon.contains(crossover_region):
        crossover_region = boundary_polygon.intersection(crossover_region)

    # Compute actual area proportion
    crossover_area = crossover_region.area
    crossover_proportion = crossover_area / total_site_area

    # Debugging output: Print the area ratio
    print(f"Crossover Region Area: {crossover_area:.2f} m²")
    print(f"Total Sheringham Shoal Site Area: {total_site_area:.2f} m²")
    print(f"Proportion of Site Used for Crossover: {crossover_proportion:.2%}")

    # Initialize child layout
    child_layout = {}
    used_positions = set()

    # Inherit turbines inside the crossover region from the fitter parent
    for tid, (x, y) in parent_1.items():
        if crossover_region.contains(Point(x, y)):
            child_layout[tid] = (x, y)
            used_positions.add((x, y))  # Track positions to avoid duplication

    # Fill remaining turbines from the less fit parent (outside crossover region)
    for tid, (x, y) in parent_2.items():
        if (x, y) not in used_positions and not crossover_region.contains(Point(x, y)):
            # Ensure spacing is maintained
            if all(distance.euclidean((x, y), t) >= min_spacing for t in child_layout.values()):
                new_id = max(child_layout.keys(), default=0) + 1  # Assign new unique ID
                child_layout[new_id] = (x, y)
                used_positions.add((x, y))

    return child_layout, crossover_region