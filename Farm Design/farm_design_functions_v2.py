import numpy as np
from math import pi

def jensen_single_model(v0, Ct, r0, x, z, z0):
    """
    Calculate the reduced wind speed downstream in the wake zone using Jensen's single wake model.

    Parameters:
    v0 (float): Freestream wind speed (m/s)
    Ct (float): Thrust coefficient
    r0 (float): Rotor radius (m)
    x (float): Downstream distance (m)
    z (float): Hub height (m)
    z0 (float): Surface roughness length (m)

    Returns:
    float: Reduced wind speed downstream (m/s)
    """
    # Compute the decay constant (alpha)
    alpha = 1 / (2 * np.log(z / z0))
    #print(f"alpha = {alpha}")

    # Calculate the wake radius at downstream distance x
    rx = r0 + alpha * x

    # Calculate reduced wind speed in the wake
    reduced_speed = v0 + v0 * (np.sqrt(1 - Ct) - 1) * (r0 / rx) ** 2
    return reduced_speed


import numpy as np

def multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z_0, d_ij):
    """
    Calculate the effective wind speed at turbine j due to the wake from turbine i.

    Parameters:
    v_i (float): Wind speed at turbine i (m/s)
    r_0 (float): Rotor radius of turbine i (m)
    Ct (float): Thrust coefficient of turbine i
    x_ij (float): Downstream distance between turbines i and j (m)
    z_0 (float): Surface roughness length (m)
    d_ij (float): Perpendicular distance between turbine i's wake centerline and turbine j (m)

    Returns:
    float: Effective wind speed at turbine j (m/s)
    bool: Partial shadowing flag
    bool: Complete shadowing flag
    bool: No shadowing flag
    """

    # Compute the decay constant (alpha)
    alpha = 1 / (2 * np.log(r_0 / z_0))  # Decay constant

    # Calculate the wake radius at downstream distance x_ij
    r_i_xij = r_0 + alpha * x_ij  # Wake radius at distance x_ij
    print(f"r_i_xij = {r_i_xij}")
    print(f"r_0 = {r_0}")
    print(f"d_ij = {d_ij}")

    # Initialize shadowing booleans
    partial_shadowing = False
    complete_shadowing = False
    no_shadowing = False

    # Calculate the rotor swept area of turbine j
    A_0 = np.pi * r_0**2

    # Handle complete shadowing when entire turbine j is inside the wake (d_ij + r_0 <= r_i_xij)
    if d_ij + r_0 <= r_i_xij:
        A_shadow_i = A_0  # Entire rotor is shadowed
        complete_shadowing = True
        print("Complete shadowing (entire rotor within wake): A_shadow_i = A_0")
        v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij)**2)
        return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing

    # Handle no shadowing (d_ij >= r_0 + r_i_xij)
    elif d_ij >= (r_0 + r_i_xij):  # No overlap between wake cone and turbine j
        print(f"r_0 + r_i_xij = {r_0 + r_i_xij}")
        A_shadow_i = 0  # No shadowing occurs
        no_shadowing = True
        print("No shadowing: A_shadow_i = 0")
        return v_i, partial_shadowing, complete_shadowing, no_shadowing

    # Handle partial shadowing (partial overlap)
    elif d_ij < (r_0 + r_i_xij) and d_ij > 0:
        try:
            # Calculate the height of the shadowed area (z_ij)
            term_1 = 4 * d_ij**2 * r_i_xij**2
            term_2 = (d_ij**2 - r_0**2 + r_i_xij**2)**2
            if term_1 > term_2:  # Ensure the square root is valid
                z_ij = (1 / d_ij) * np.sqrt(term_1 - term_2)
                print(f"z_ij = {z_ij}")
            else:
                z_ij = 0  # Invalid geometry for z_ij
                print("Invalid geometry for z_ij, setting z_ij = 0")
        except ValueError:
            z_ij = 0  # Handle mathematical errors safely
            print("Error in z_ij calculation, setting z_ij = 0")

        try:
            # Calculate the perpendicular distance L_ij
            L_ij = d_ij - np.sqrt(r_0**2 - (z_ij / 2)**2)
            print(f"L_ij = {L_ij}")
        except ValueError:
            L_ij = 0  # Handle invalid geometry
            print("Error in L_ij calculation, setting L_ij = 0")

        # Calculate the shadowed area A_shadow_i
        try:
            if L_ij > 0:  # Ensure a valid overlap exists
                A_shadow_i = (
                    r_i_xij**2 * np.arccos(L_ij / r_i_xij) +
                    r_0**2 * np.arccos((d_ij - L_ij) / r_i_xij) -
                    d_ij * z_ij
                )
                partial_shadowing = True
                print("Partial shadowing: A_shadow_i calculated based on overlap area")
            else:
                A_shadow_i = 0  # No shadowing occurs
                no_shadowing = True
                print("No shadowing: A_shadow_i = 0")
        except ValueError:
            A_shadow_i = 0  # Handle invalid geometry
            print("Error in A_shadow_i calculation, setting A_shadow_i = 0")

        # Calculate the effective wind speed at turbine j
        v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij)**2 * (A_shadow_i / A_0))

        # Return effective wind speed and shadowing flags
        return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing
        

