import farm_design_functions_v1 as fd
import numpy as np
import matplotlib.pyplot as plt

# Input parameters from the first iteration of BEM for the 20MW turbine
v0 = 9.519  # Freestream wind speed (m/s)
Ct = 0.761  # Thrust coefficient
r0 = 126.0  # Rotor radius (m)
z = 175.0   # Hub height (m)
z0 = 7.8662884488e-04  # Surface roughness length (m)

# Rotor diameter
D = 2 * r0  # Rotor diameter (m)

# Range of downstream distances up to 20 rotor diameters
x_values = np.linspace(0.5*D, 20 * D, 1000)  # From 1 rotor diameter to 15 rotor diameters
x_rotor_diameters = x_values / D  # Normalize by rotor diameter

# Print an example reduced wind speed at a specific x
x_test = 5 * D  # Test downstream distance (e.g., 5 rotor diameters)
reduced_speed_test = fd.jensen_single_model(v0, Ct, r0, x_test, z, z0)
print(f"Reduced wind speed at x = {x_test:.1f} m downstream ({x_test/D:.1f} rotor diameters): {reduced_speed_test:.3f} m/s")

# Calculate reduced wind speed for each x
reduced_speeds = [fd.jensen_single_model(v0, Ct, r0, x, z, z0) for x in x_values]

# Calculate wind speed deficit for each x
speed_deficits = [1 - (v / v0) for v in reduced_speeds]

# --- Styled Plot for Reduced Wind Speed ---

# Set consistent style
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})

fig, ax = plt.subplots(figsize=(8, 5))

# Plot reduced speed
ax.plot(
    x_rotor_diameters,
    reduced_speeds,
    label="Reduced Wind Speed (Jensen's Model)",
    color="blue",
    linewidth=1.5
)

# Plot upstream wind speed line
ax.axhline(
    y=v0,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=fr"Upstream Wind Speed $v_0$ = {v0:.3f} m/s"
)

# Axes labels and title
ax.set_xlabel(r"Downstream Distance, $x$ (Rotor Diameters)")
ax.set_ylabel(r"Wind Speed, $v$ (m/s)")
#ax.set_title(r"Reduced Wind Speed vs Downstream Distance (Jensen's Model)")

# Grid and ticks
ax.grid(True, linestyle='-', linewidth=0.7, alpha=0.7)
ax.tick_params(axis='both', labelsize=12)

# Legend
ax.legend(prop={"family": "Arial", "size": 14})

plt.tight_layout()
plt.show()