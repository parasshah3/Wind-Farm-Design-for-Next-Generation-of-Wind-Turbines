import numpy as np
from math import pi

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

    try:
        # Compute the decay constant (alpha)
        alpha = 0.5 / np.log(z / z_0)  # Decay constant

        # Calculate the wake radius at downstream distance x_ij
        r_i_xij = r_0 + alpha * x_ij  # Wake radius at distance x_ij
        #print(f"r_i_xij = {r_i_xij}")
        #print(f"r_0 = {r_0}")
        #print(f"d_ij = {d_ij}")

        # Calculate the rotor swept area of turbine j
        A_0 = np.pi * r_0**2

        # Case 1: Complete shadowing (entire rotor within the wake)
        if d_ij + r_0 <= r_i_xij:
            A_shadow_i = A_0  # Entire rotor is shadowed
            complete_shadowing = True
            print("Complete shadowing (entire rotor within wake): A_shadow_i = A_0")
            v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij)**2)
            print(f"v_j_xij = {v_j_xij}")
            print(f"d_ij = {d_ij}")
            return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i

        # Case 2: No shadowing (no overlap between wake and turbine rotor)
        elif d_ij >= (r_0 + r_i_xij):
            #print(f"r_0 + r_i_xij = {r_0 + r_i_xij}")
            A_shadow_i = 0  # No shadowing occurs
            no_shadowing = True
            print("No shadowing: A_shadow_i = 0")
            return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i

        # Case 3: Partial shadowing (partial overlap)
        elif d_ij < (r_0 + r_i_xij):
            #try:
            # Calculate the height of the shadowed area (z_ij)
            term_1 = 4 * d_ij**2 * r_i_xij**2
            term_2 = (d_ij**2 - r_0**2 + r_i_xij**2)**2
            if term_1 > term_2:  # Ensure the square root is valid
                z_ij = (1 / d_ij) * np.sqrt(term_1 - term_2)
                #print(f"z_ij = {z_ij}")
            else:
                z_ij = 0  # Invalid geometry for z_ij
                print("Invalid geometry for z_ij, setting z_ij = 0")
            #except ValueError:
             #   z_ij = 0  # Handle mathematical errors safely
               # print("Error in z_ij calculation, setting z_ij = 0")

            #try:
            # Calculate the perpendicular distance L_ij
            if z_ij > 0:
                L_ij = abs(d_ij - np.sqrt(r_0**2 - (z_ij / 2)**2))
                #print(f"L_ij = {L_ij}")
            else:
                L_ij = 0
           #except ValueError:
               # L_ij = 0  # Handle invalid geometry
               # print("Error in L_ij calculation, setting L_ij = 0")

            # Calculate the shadowed area A_shadow_i
        #try:
            if L_ij > 0:  # Ensure a valid overlap exists
                #A_shadow_i = (
                #    r_i_xij**2 * np.arccos(L_ij / r_i_xij) +
                #    r_0**2 * np.arccos((d_ij - L_ij) / r_0) -
                #    0.5*d_ij * z_ij
                # )
                A_shadow_i = (
                    r_0**2 * np.arccos((d_ij**2 + r_0**2 - r_i_xij**2) / (2 * d_ij * r_0)) +
                    r_i_xij**2 * np.arccos((d_ij**2 + r_i_xij**2 - r_0**2) / (2 * d_ij * r_i_xij)) -
                    0.5 * d_ij * z_ij
                ) 
                partial_shadowing = True
                print("Partial shadowing: A_shadow_i calculated based on overlap area")
                # Calculate the effective wind speed at turbine j
                v_j_xij = v_i * (1 - (1 - np.sqrt(1 - Ct)) * (r_0 / r_i_xij)**2 * (A_shadow_i / A_0))
                return v_j_xij, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i
            else:
                A_shadow_i = 0  # No shadowing occurs
                no_shadowing = True
                print("No shadowing via error: A_shadow_i = 0")
                if d_ij + r_0 <= r_i_xij:
                    print('d_ij + r_0 <= r_i_xij is True')
                #print(f"A_shadow_i = {A_shadow_i}")
                #print(f"z_ij = {z_ij}")
                #print(f"d_ij = {d_ij}")
                #print(f"r_0: {r_0}")
                #print(f"d_ij + r_0: {d_ij + r_0}")
                #print(f"r_i_xij: {r_i_xij}")
                #print(f"v_j_xij = {v_i}")
                return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i
        #except ValueError:
               # A_shadow_i = 0  # Handle invalid geometry
              #  print("Error in A_shadow_i calculation, setting A_shadow_i = 0")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Return default values if an error occurred
    return v_i, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i