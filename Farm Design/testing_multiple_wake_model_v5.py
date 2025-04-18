import numpy as np
import matplotlib.pyplot as plt
import farm_design_functions_v5 as fd

# Input parameters for the test case
v_i = 9.519  # Upstream wind speed (m/s)
Ct = 0.761  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)
z = 100  # Hub height (m)
x_ij = 2 * 2 * r_0  # Downstream distance (m)

# Compute r_i_xij
alpha = 0.5 / np.log(z / z_0)  # Decay constant
r_i_xij = r_0 + alpha * x_ij  # Wake radius at downstream distance x_ij

# Define dij_values
dij_values = np.linspace(0, 2 * (r_0 + r_i_xij), 500)

# Convert dij_values to rotor diameters for plotting
dij_rotor_diameters = dij_values / (2 * r_0)

# Arrays for outputs
vj_values = []
Lij_values = []
zij_values = []
A_shadow_normalized = []  # Shadow area normalized by A_0

# Transition points
complete_to_partial_dij = None
partial_to_no_shadow_dij = None

# States for shadowing
previous_shadowing_state = None

for d_ij in dij_values:
    v_j, partial_shadowing, complete_shadowing, no_shadowing, L_ij, z_ij, A_shadow_i = fd.multiple_wake_model_ij(
        v_i, r_0, Ct, x_ij, z, z_0, d_ij
    )
    vj_values.append(v_j)
    Lij_values.append(L_ij)
    zij_values.append(z_ij)

    # Normalize A_shadow by A_0 (turbine swept area)
    A_0 = np.pi * r_0**2
    A_shadow_normalized.append(A_shadow_i / A_0 if A_shadow_i is not None else 0)

    # Determine the current shadowing state
    if complete_shadowing:
        current_shadowing_state = "complete"
    elif partial_shadowing:
        current_shadowing_state = "partial"
    elif no_shadowing:
        current_shadowing_state = "none"

    # Detect transitions
    if previous_shadowing_state is None:
        previous_shadowing_state = current_shadowing_state

    if previous_shadowing_state == "complete" and current_shadowing_state == "partial" and complete_to_partial_dij is None:
        complete_to_partial_dij = d_ij
    elif previous_shadowing_state == "partial" and current_shadowing_state == "none" and partial_to_no_shadow_dij is None:
        partial_to_no_shadow_dij = d_ij

    previous_shadowing_state = current_shadowing_state

# Convert transition points to rotor diameters
complete_to_partial_dij_rotor = complete_to_partial_dij / (2 * r_0) if complete_to_partial_dij else None
partial_to_no_shadow_dij_rotor = partial_to_no_shadow_dij / (2 * r_0) if partial_to_no_shadow_dij else None

import matplotlib.ticker as ticker

# --- Standard Formatting ---
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})

fig, ax = plt.subplots(figsize=(8, 6))

# Plot Effective Wind Speed at Turbine j vs d_ij
ax.plot(dij_rotor_diameters, vj_values, label=r'Effective Wind Speed at Turbine $j$', 
         linewidth=1.5)

# Plot upstream wind speed line
ax.axhline(v_i, color='red', linestyle='--', linewidth=1.2, label=r'Upstream Wind Speed $v_i$')

# Plot shadowing transition lines
if complete_to_partial_dij_rotor:
    ax.axvline(complete_to_partial_dij_rotor, color='black', linestyle=':', linewidth=1.5, 
               label=r'$ \text{Complete} \rightarrow \text{Partial Shadowing} $')
if partial_to_no_shadow_dij_rotor:
    ax.axvline(partial_to_no_shadow_dij_rotor, color='orange', linestyle=':', linewidth=1.5, 
               label=r'$ \text{Partial} \rightarrow \text{No Shadowing} $')

# Axis labels and title
ax.set_xlabel(r'Perpendicular Distance, $d_{ij}$ (Rotor Diamters)', fontsize=14)
ax.set_ylabel(r'Effective Wind Speed, $v_j$ (m/s)', fontsize=14)
#ax.set_title(r'Effective Wind Speed $v_j$ vs $d_{ij}/D$ (Downstream Distance: $2D$)', fontsize=16, fontweight='bold')

# Grid and ticks
ax.grid(True, linestyle='-', linewidth=0.6, alpha=0.6)
ax.tick_params(axis='both', labelsize=12)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(-3, 4), useMathText=True)
ax.yaxis.offsetText.set_fontsize(0)
ax.yaxis.offsetText.set_fontname("Arial")

# Legend
ax.legend(loc='best', prop={"family": "Arial", "size": 14})

plt.tight_layout()
plt.show()

# Code for other plots (currently commented out):

# # Plot A_shadow (normalized by A_0) vs dij
# axes[0, 1].plot(dij_rotor_diameters, A_shadow_normalized, label=r"$A_{\text{shadow},i} / A_0$", color="orange")
# if complete_to_partial_dij_rotor:
#     axes[0, 1].axvline(complete_to_partial_dij_rotor, color="blue", linestyle="--", label="Complete to Partial Shadowing")
# if partial_to_no_shadow_dij_rotor:
#     axes[0, 1].axvline(partial_to_no_shadow_dij_rotor, color="orange", linestyle="--", label="Partial to No Shadowing")
# axes[0, 1].set_xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
# axes[0, 1].set_ylabel(r"Normalized Shadow Area $A_{\text{shadow},i} / A_0$")
# axes[0, 1].set_title("Normalized Shadow Area $A_{\text{shadow},i}$ vs. $d_{ij}$ (Downstream Distance: 2 Rotor Diameters)")
# axes[0, 1].legend()
# axes[0, 1].grid()

# # Plot Lij vs dij
# axes[1, 0].plot(dij_rotor_diameters, Lij_values, label=r"$L_{ij}$", color="green")
# if complete_to_partial_dij_rotor:
#     axes[1, 0].axvline(complete_to_partial_dij_rotor, color="blue", linestyle="--", label="Complete to Partial Shadowing")
# if partial_to_no_shadow_dij_rotor:
#     axes[1, 0].axvline(partial_to_no_shadow_dij_rotor, color="orange", linestyle="--", label="Partial to No Shadowing")
# axes[1, 0].set_xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
# axes[1, 0].set_ylabel(r"$L_{ij}$ (m)")
# axes[1, 0].set_title(r"$L_{ij}$ vs. $d_{ij}$ (Downstream Distance: 2 Rotor Diameters)")
# axes[1, 0].legend()
# axes[1, 0].grid()

# # Plot zij vs dij
# axes[1, 1].plot(dij_rotor_diameters, zij_values, label=r"$z_{ij}$", color="purple")
# if complete_to_partial_dij_rotor:
#     axes[1, 1].axvline(complete_to_partial_dij_rotor, color="blue", linestyle="--", label="Complete to Partial Shadowing")
# if partial_to_no_shadow_dij_rotor:
#     axes[1, 1].axvline(partial_to_no_shadow_dij_rotor, color="orange", linestyle="--", label="Partial to No Shadowing")
# axes[1, 1].set_xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
# axes[1, 1].set_ylabel(r"$z_{ij}$ (m)")
# axes[1, 1].set_title(r"$z_{ij}$ vs. $d_{ij}$ (Downstream Distance: 2 Rotor Diameters)")
# axes[1, 1].legend()
# axes[1, 1].grid()