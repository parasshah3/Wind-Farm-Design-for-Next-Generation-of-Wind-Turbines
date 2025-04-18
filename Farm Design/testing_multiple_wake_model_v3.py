import numpy as np
import matplotlib.pyplot as plt
import farm_design_functions_v2 as fd

# Input parameters for the test case
v_i = 9.519  # Upstream wind speed (m/s)
Ct = 0.761  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)
x_ij = 1 * 2 * r_0  # Downstream distance (m) (1 rotor diameter)

# Compute r_i_xij
alpha = 1 / (2 * np.log(r_0 / z_0))  # Decay constant
r_i_xij = r_0 + alpha * x_ij  # Wake radius at downstream distance x_ij

# Compute the critical value of d_ij where no shadowing occurs
dij_no_shadow = r_0 + r_i_xij

# Define dij_values as a range of perpendicular distances (in meters)
dij_values = np.linspace(0, 2 * dij_no_shadow, 500)  # Extend beyond the no-shadowing threshold

# Arrays to store results
Lij_values = []
zij_values = []
A_shadow_values = []
vj_values = []

# Transition points
complete_to_partial_dij = None
partial_to_no_shadow_dij = None

for d_ij in dij_values:
    # Calculate effective wind speed and shadowing conditions
    v_j, partial_shadowing, complete_shadowing, no_shadowing = fd.multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z_0, d_ij)
    vj_values.append(v_j)

    # Identify transitions
    if complete_shadowing and complete_to_partial_dij is None:
        complete_to_partial_dij = d_ij
    elif partial_shadowing and not complete_shadowing and partial_to_no_shadow_dij is None:
        partial_to_no_shadow_dij = d_ij

    try:
        # Calculate z_ij
        term_1 = 4 * d_ij**2 * r_i_xij**2
        term_2 = (d_ij**2 - r_0**2 + r_i_xij**2)**2
        if term_1 >= term_2:  # Ensure the square root is valid
            z_ij = (1 / d_ij) * np.sqrt(term_1 - term_2) if d_ij != 0 else 0
        else:
            z_ij = 0  # No intersection
    except ValueError:
        z_ij = 0  # Handle invalid geometry

    try:
        # Calculate L_ij
        if z_ij > 0:
            L_ij = d_ij - np.sqrt(r_0**2 - (z_ij / 2)**2)
        else:
            L_ij = 0  # No overlap
    except ValueError:
        L_ij = 0  # Handle invalid geometry

    try:
        # Calculate A_shadow
        if L_ij > 0:
            A_shadow_i = (
                r_i_xij**2 * np.arccos(L_ij / r_i_xij) +
                r_0**2 * np.arccos((d_ij - L_ij) / r_0) -
                d_ij * z_ij
            )
        else:
            A_shadow_i = 0  # No shadowing
    except ValueError:
        A_shadow_i = 0  # Handle invalid geometry

    Lij_values.append(L_ij)
    zij_values.append(z_ij)
    A_shadow_values.append(A_shadow_i)

# Convert perpendicular distances to rotor diameters
dij_rotor_diameters = dij_values / (2 * r_0)
complete_to_partial_dij_rotor = complete_to_partial_dij / (2 * r_0) if complete_to_partial_dij else None
partial_to_no_shadow_dij_rotor = partial_to_no_shadow_dij / (2 * r_0) if partial_to_no_shadow_dij else None

# Create the plots in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot Lij vs dij
axes[0, 0].plot(dij_rotor_diameters, Lij_values, label=r"$L_{ij}$", color="green")
axes[0, 0].axvline(dij_no_shadow / (2 * r_0), color="red", linestyle="--", label=r"No Shadowing Threshold $d_{ij} = r_0 + r_i(x_{ij})$")
if complete_to_partial_dij_rotor:
    axes[0, 0].axvline(complete_to_partial_dij_rotor, color="blue", linestyle="--", label="Transition: Complete to Partial Shadowing")
if partial_to_no_shadow_dij_rotor:
    axes[0, 0].axvline(partial_to_no_shadow_dij_rotor, color="orange", linestyle="--", label="Transition: Partial to No Shadowing")
axes[0, 0].set_xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
axes[0, 0].set_ylabel(r"$L_{ij}$ (m)")
axes[0, 0].set_title(r"$L_{ij}$ vs. $d_{ij}$")
axes[0, 0].legend()
axes[0, 0].grid()

# Plot zij vs dij
axes[0, 1].plot(dij_rotor_diameters, zij_values, label=r"$z_{ij}$", color="purple")
axes[0, 1].axvline(dij_no_shadow / (2 * r_0), color="red", linestyle="--", label=r"No Shadowing Threshold $d_{ij} = r_0 + r_i(x_{ij})$")
if complete_to_partial_dij_rotor:
    axes[0, 1].axvline(complete_to_partial_dij_rotor, color="blue", linestyle="--", label="Transition: Complete to Partial Shadowing")
if partial_to_no_shadow_dij_rotor:
    axes[0, 1].axvline(partial_to_no_shadow_dij_rotor, color="orange", linestyle="--", label="Transition: Partial to No Shadowing")
axes[0, 1].set_xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
axes[0, 1].set_ylabel(r"$z_{ij}$ (m)")
axes[0, 1].set_title(r"$z_{ij}$ vs. $d_{ij}$")
axes[0, 1].legend()
axes[0, 1].grid()

# Plot Ashadow vs dij
axes[1, 0].plot(dij_rotor_diameters, A_shadow_values, label=r"$A_{\text{shadow},i}$", color="orange")
axes[1, 0].axvline(dij_no_shadow / (2 * r_0), color="red", linestyle="--", label=r"No Shadowing Threshold $d_{ij} = r_0 + r_i(x_{ij})$")
if complete_to_partial_dij_rotor:
    axes[1, 0].axvline(complete_to_partial_dij_rotor, color="blue", linestyle="--", label="Transition: Complete to Partial Shadowing")
if partial_to_no_shadow_dij_rotor:
    axes[1, 0].axvline(partial_to_no_shadow_dij_rotor, color="orange", linestyle="--", label="Transition: Partial to No Shadowing")
axes[1, 0].set_xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
axes[1, 0].set_ylabel(r"$A_{\text{shadow},i}$ (m²)")
axes[1, 0].set_title(r"$A_{\text{shadow},i}$ vs. $d_{ij}$")
axes[1, 0].legend()
axes[1, 0].grid()

# Plot vj vs dij
axes[1, 1].plot(dij_rotor_diameters, vj_values, label=r"$v_j$", color="blue")
axes[1, 1].axhline(v_i, color="red", linestyle="--", label="Upstream Wind Speed $v_i$")
axes[1, 1].axvline(dij_no_shadow / (2 * r_0), color="red", linestyle="--", label=r"No Shadowing Threshold $d_{ij} = r_0 + r_i(x_{ij})$")
if complete_to_partial_dij_rotor:
    axes[1, 1].axvline(complete_to_partial_dij_rotor, color="blue", linestyle="--", label="Transition: Complete to Partial Shadowing")
if partial_to_no_shadow_dij_rotor:
    axes[1, 1].axvline(partial_to_no_shadow_dij_rotor, color="orange", linestyle="--", label="Transition: Partial to No Shadowing")
axes[1, 1].set_xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
axes[1, 1].set_ylabel(r"$v_j$ (m/s)")
axes[1, 1].set_title(r"$v_j$ vs. $d_{ij}$")
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()