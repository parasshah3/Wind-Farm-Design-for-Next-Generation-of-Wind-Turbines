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