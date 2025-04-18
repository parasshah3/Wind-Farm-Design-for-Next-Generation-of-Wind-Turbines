import numpy as np
import matplotlib.pyplot as plt
import farm_design_functions_v5 as fd

# Input parameters
v_i = 9.519  # Freestream wind speed (m/s)
Ct = 0.761  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)
z = 100  # Hub height (m)
recovery_threshold = 0.9 * v_i  # 90% recovery threshold

# Define range of perpendicular distances (dij) to test
dij_values = np.linspace(0, 5 * r_0, 50)  # From 0 to 5 rotor radii

# Arrays to store results for both plots
xij_required = []  # Downstream distance required for recovery
vj_values = []  # Effective wind speed values for varying dij

# **Plot 1: Downstream Distance vs Perpendicular Distance**
for d_ij in dij_values:
    xij_values = np.linspace(2 * r_0, 20 * r_0, 500)  # Test downstream distances
    recovery_xij = None

    # Calculate xij for recovery
    for x_ij in xij_values:
        v_j, _, _, _, _, _, _ = fd.multiple_wake_model_ij(v_i, r_0, Ct, x_ij, z, z_0, d_ij)
        if v_j >= recovery_threshold:
            recovery_xij = x_ij
            break

    # Append recovery distance or NaN if not found
    xij_required.append(recovery_xij / r_0 if recovery_xij else np.nan)

# **Plot 2: Effective Wind Speed vs Perpendicular Distance**
x_ij_fixed = 2 * 2 * r_0  # Fixed downstream distance
for d_ij in dij_values:
    v_j, _, _, _, _, _, _ = fd.multiple_wake_model_ij(v_i, r_0, Ct, x_ij_fixed, z, z_0, d_ij)
    vj_values.append(v_j)

# Convert d_ij to rotor diameters for plotting
dij_rotor_diameters = dij_values / (2 * r_0)

# Plot Downstream Distance vs Perpendicular Distance
plt.figure(figsize=(10, 6))
plt.plot(dij_rotor_diameters, xij_required, label=r"$x_{ij}$ for 90% $v_0$ Recovery", color="blue", marker="o")
plt.xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
plt.ylabel(r"Downstream Distance $x_{ij}$ [Rotor Diameters]")
plt.title("Downstream Distance $x_{ij}$ Required for 90% $v_0$ Recovery vs. $d_{ij}$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Plot Effective Wind Speed vs Perpendicular Distance
plt.figure(figsize=(10, 6))
plt.plot(dij_rotor_diameters, vj_values, label=r"$v_j(x_{ij} = 4r_0)$", color="green", linestyle="--", marker="x")
plt.xlabel(r"Perpendicular Distance $d_{ij}$ [Rotor Diameters]")
plt.ylabel(r"Effective Wind Speed $v_j$ [m/s]")
plt.title("Effective Wind Speed $v_j$ vs. Perpendicular Distance $d_{ij}$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()