import farm_design_functions_v8 as fd  # Import the module with your functions
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the test (all in metres)
Ct = 0.806  # Thrust coefficient
r_0 = 126.0  # Rotor radius (m)
D = 2 * r_0  # Rotor diameter (m)
z = 140  # Hub height (m)
z_0 = 7.8662884488e-04  # Surface roughness length (m)

# Test 1: Original arrangement with 7 turbines
A = 1.5 * D  # Scaling factor for turbine spacing
turbine_positions_test1 = {
    1: (0 * A, 0 * A),
    2: (1 * A, 0.5 * A),
    3: (1 * A, -0.5 * A),
    4: (2 * A, 0 * A),
    5: (3 * A, 0.5 * A),
    6: (3 * A, -0.5 * A),
    7: (4 * A, 0 * A),
}

# Test 2: 7 turbines in a straight line
turbine_positions_test2 = {i + 1: (i * 1.5 * D, 0) for i in range(7)}

def run_test_and_plot(turbine_positions, test_label):
    """
    Run the multiple wake model test and plot the turbine arrangement with Weibull parameters.

    Parameters:
    turbine_positions (dict): Dictionary with turbine IDs and (x, y) positions
    test_label (str): Label for the test case
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    # Plot style
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14
    })

    # Constants
    results = fd.mwm_identical_turbines_WD(Ct, r_0, z, z_0, turbine_positions)
    D = 2 * r_0  # Rotor diameter

    print(f"\nResults from `mwm_identical_turbines_WD` for {test_label}:")
    for turbine_id, data in results.items():
        print(f"Turbine {turbine_id}: k = {data['k']:.4f}, λ = {data['lambda']:.4f}")

    # Normalised coordinates
    x_coords = [pos[0] / D for pos in turbine_positions.values()]
    y_coords = [pos[1] / D for pos in turbine_positions.values()]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    x_pad, y_pad = 0.5, 0.5
    x_lim = (x_min - x_pad, x_max + x_pad)
    y_lim = (y_min - y_pad, y_max + y_pad)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_coords, y_coords, s=50, label="Turbines")

    for turbine_id, (x, y) in turbine_positions.items():
        k = results[turbine_id]['k']
        λ = results[turbine_id]['lambda']
        ax.text(x / D, y / D + 0.1, f"{turbine_id}\n(k={k:.3f}, λ={λ:.3f})",
                fontsize=11, ha="center", color="black", fontname="Arial")

    # Wind arrows on left
    for y_arrow in np.arange(y_lim[0], y_lim[1] + 0.01, 0.55):
        ax.arrow(x_lim[0] - 0.2, y_arrow, 0.6, 0,
                 head_width=0.1, head_length=0.15, fc='green', ec='green', alpha=0.7)

    # Axes and formatting
    #ax.set_title(f"{test_label} – Turbine Layout with Weibull Parameters", fontsize=16, fontweight='bold')
    ax.set_xlabel("Downstream Distance, $x$ (Rotor Diameters)", fontsize=13)
    ax.set_ylabel("Perpendicular Distance, $d$ (Rotor Diameters)", fontsize=13)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='-', linewidth=0.6, alpha=0.6)
    ax.tick_params(axis='both', labelsize=12)

    # Scientific format on ticks
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(-3, 4), useMathText=True)
    ax.xaxis.offsetText.set_fontsize(10)
    ax.yaxis.offsetText.set_fontsize(10)
    ax.xaxis.offsetText.set_fontname("Arial")
    ax.yaxis.offsetText.set_fontname("Arial")

    plt.tight_layout()
    plt.show()

# Run Test 1
run_test_and_plot(turbine_positions_test1, "Original Arrangement")

# Run Test 2
run_test_and_plot(turbine_positions_test2, "Straight Line Arrangement")