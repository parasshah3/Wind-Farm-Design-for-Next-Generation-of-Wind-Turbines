import farm_design_functions_v7 as fd  # Import the module with your functions
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the test (all in metres)
v_i = 9.399 # Freestream wind speed (m/s)
sigma_0 = 2.436  # Freestream standard deviation (m/s)
Ct = 0.761  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
D = 2 * r_0  # Rotor diameter (m)
z = 100  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# Test 1: Original arrangement with 7 turbines
A = 1.5 * D  # Scaling factor for turbine spacing
turbine_positions_test1 = {
    1: (0 * A, 0 * A),    # Turbine 1
    2: (1 * A, 0.5 * A),  # Turbine 2
    3: (1 * A, -0.5 * A), # Turbine 3
    4: (2 * A, 0 * A),    # Turbine 4
    5: (3 * A, 0.5 * A),  # Turbine 5
    6: (3 * A, -0.5 * A), # Turbine 6
    7: (4 * A, 0 * A),    # Turbine 7
}

# Test 2: 7 turbines in a straight line
turbine_positions_test2 = {
    i + 1: (i * 1.5 * D, 0) for i in range(7)  # Turbines evenly spaced in a straight line
}

def run_test_and_plot(turbine_positions, test_label):
    """
    Run the multiple wake model test and plot the turbine arrangement.

    Parameters:
    turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions
    test_label (str): Label for the test case
    """
    # Calculate effective wind speeds and standard deviations using `mwm_identical_turbines_speed_sd`
    results = fd.mwm_identical_turbines_speed_sd(v_i, sigma_0, Ct, r_0, z, z_0, turbine_positions)

    # Print results for each turbine
    print(f"\nResults from `mwm_identical_turbines_speed_sd` for {test_label}:")
    for turbine_id, data in results.items():
        print(f"Turbine {turbine_id}: v_j = {data['v_j']:.4f} m/s, sigma_j = {data['sigma_j']:.4f} m/s")

    # Plot the turbine arrangement
    x_coords = [pos[0] / D for pos in turbine_positions.values()]  # Convert to rotor diameters for plotting
    y_coords = [pos[1] / D for pos in turbine_positions.values()]

    # Determine dynamic plot limits
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Add padding to plot limits
    padding = 0.5  # Padding in rotor diameters
    x_lim = (x_min - padding, x_max + padding)
    y_lim = (y_min - padding, y_max + padding)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x_coords, y_coords, color="black", s=100, label="Turbines")

    # Add labels for each turbine including calculated effective wind speed and standard deviation
    for turbine_id, (x, y) in turbine_positions.items():
        v_j = results[turbine_id]['v_j']
        sigma_j = results[turbine_id]['sigma_j']
        ax.text(x / D, y / D + 0.2, f"{turbine_id} (v_j={v_j:.4f}, σ_j={sigma_j:.4f})", fontsize=10, ha="center", color="blue")

    # Add incoming wind arrows (v0)
    for y_arrow in np.arange(y_lim[0], y_lim[1], 0.9):  # Adjust range dynamically
        ax.arrow(x_lim[0] + 0.1, y_arrow, 0.4, 0, head_width=0.1, head_length=0.1, fc="black", ec="black")

    # Add v0 and sigma_0 labels near the arrows
    ax.text(x_lim[0], y_lim[1] + 0.1, f"$v_0 = {v_i:.4f}$ m/s\n$σ_0 = {sigma_0:.4f}$ m/s", fontsize=12, color="black")

    # Formatting
    ax.set_title(f"Wind Turbine Arrangement ({test_label})", fontsize=14)
    ax.set_xlabel("Downstream Distance (Rotor Diameters)", fontsize=12)
    ax.set_ylabel("Perpendicular Distance (Rotor Diameters)", fontsize=12)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)  # Horizontal axis
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)  # Vertical axis
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")
    plt.legend(loc="upper left", fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.show()

# Run Test 1
run_test_and_plot(turbine_positions_test1, "Original Arrangement")

# Run Test 2
run_test_and_plot(turbine_positions_test2, "Straight Line Arrangement")