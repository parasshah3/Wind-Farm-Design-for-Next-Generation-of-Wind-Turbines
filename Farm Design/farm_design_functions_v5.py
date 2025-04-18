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

def multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij):
    """
    Calculate the effective wind speed at turbine j due to the wake from turbine i,
    and return intermediate parameters.

    Parameters:
    v_i (float): Wind speed at turbine i (m/s)
    r_0 (float): Rotor radius of turbine i
    Ct (float): Thrust coefficient of turbine i
    x_ij (float): Downstream distance between turbines i and j (m)
    z (float): Hub height (m)
    z_0 (float): Surface roughness length (m)
    d_ij (float): Perpendicular distance between turbine i's wake centerline and turbine j (m)

    Returns:
    tuple: (v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i)
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

    # Calculate the rotor swept area of turbine j
    A_0 = np.pi * r_0**2

    # Case 1: Complete shadowing (entire rotor within the wake)
    if d_ij + r_0 <= r_i_xij:
        A_shadow_i = A_0  # Entire rotor is shadowed
        complete_shadowing = True
        print("Case: Complete shadowing")
        v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij)**2)
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
        term_2 = (d_ij**2 - r_0**2 + r_i_xij**2)**2
        z_ij = (1 / d_ij) * np.sqrt(term_1 - term_2) if term_1 > term_2 else 0

        # Calculate the perpendicular distance L_ij
        L_ij = abs(d_ij - np.sqrt(r_0**2 - (z_ij / 2)**2)) if z_ij > 0 else 0

        # Calculate the shadowed area A_shadow_i
        if L_ij > 0:
            try:
                A_shadow_i = (
                    r_0**2 * np.arccos((d_ij**2 + r_0**2 - r_i_xij**2) / (2 * d_ij * r_0)) +
                    r_i_xij**2 * np.arccos((d_ij**2 + r_i_xij**2 - r_0**2) / (2 * d_ij * r_i_xij)) -
                    0.5 * d_ij * z_ij
                )
                partial_shadowing = True
                v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij)**2 * (A_shadow_i / A_0))
                return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i
            except ValueError:
                A_shadow_i = 0  # Handle invalid geometry
        else:
            A_shadow_i = 0  # No valid overlap

    # Default return for no valid overlap or partial shadowing calculation errors
    no_shadowing = True
    print("Case: No valid overlap (default case)")
    return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i


