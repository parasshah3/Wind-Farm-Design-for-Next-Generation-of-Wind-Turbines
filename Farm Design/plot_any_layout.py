import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_wind_farm_layout(layout_dict, boundary_coords, title="Wind Farm Layout", annotate=True):
    """
    Plots the wind farm layout and boundary with consistent formatting.

    Parameters:
    - layout_dict (dict): {turbine_id: (x, y)} dictionary.
    - boundary_coords (np.ndarray): Nx2 array of UTM boundary coordinates.
    - title (str): Plot title.
    - annotate (bool): Whether to annotate turbine IDs.
    """

    if not layout_dict:
        print("Layout is empty. Nothing to plot.")
        return

    # Global style settings
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14
    })

    # Extract turbine coordinates and IDs
    coords = np.array(list(layout_dict.values()))
    ids = list(layout_dict.keys())

    # Plot setup
    plt.figure(figsize=(8, 8))
    plt.plot(boundary_coords[:, 0], boundary_coords[:, 1], color='black', linewidth=0.85, alpha=0.7, label="Site Boundary")
    plt.scatter(coords[:, 0], coords[:, 1], s=15, label="Turbines")

    # Annotate turbines
    if annotate:
        for tid, (x, y) in zip(ids, coords):
            plt.text(x, y + 70, str(tid), fontsize=10, ha='center', color='black', fontname="Arial")

    # Axes limits with padding
    all_x = np.concatenate([boundary_coords[:, 0], coords[:, 0]])
    all_y = np.concatenate([boundary_coords[:, 1], coords[:, 1]])
    x_pad, y_pad = 1000, 1000
    plt.xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    plt.ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    # Labels, grid, and formatting
    plt.xlabel("Easting (m)", fontsize=14, fontname="Arial")
    plt.ylabel("Northing (m)", fontsize=14, fontname="Arial")
    #plt.title(title, fontsize=16, fontweight='bold', fontname="Arial")
    plt.legend(prop={"family": "Arial", "size": 14})
    plt.xticks(fontsize=12, fontname="Arial")
    plt.yticks(fontsize=12, fontname="Arial")
    plt.grid(True, linestyle='-', linewidth=0.7, alpha=0.7)
    plt.axis("equal")

    # Scientific axis formatting with corner multiplier
    ax = plt.gca()
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(-3, 4), useMathText=True)
    ax.xaxis.offsetText.set_fontsize(10)
    ax.yaxis.offsetText.set_fontsize(10)
    ax.xaxis.offsetText.set_fontname("Arial")
    ax.yaxis.offsetText.set_fontname("Arial")

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example boundary (Sheringham Shoal)
    boundary_utm = np.array([
        (371526.684, 5893424.206),
        (378450.513, 5890464.698),
        (380624.287, 5884484.707),
        (373690.373, 5887454.043),
        (371526.684, 5893424.206)  # Close loop
    ])

    # Example layout
    example_layout = {3: (376485.4328840963, 5889819.546359819), 5: (377370.96910128876, 5886997.719520554), 14: (376099.60559437465, 5888684.997824693), 15: (377008.14704147686, 5887741.535986626), 41: (376489.3632228575, 5886261.012067808), 42: (377820.5243191548, 5889215.4192167455), 43: (378888.0127771652, 5888153.345807371), 48: (378187.11628246243, 5890424.9291482465), 49: (379336.22907238384, 5886758.931117519), 50: (379609.47304187826, 5885027.189475718), 51: (375670.10269893217, 5890619.632652296), 53: (375316.0399097486, 5891688.045271524), 57: (378887.4952697866, 5885639.270641662), 58: (374210.0134742433, 5889128.762061635), 59: (373198.8671164582, 5888866.212132453), 60: (372438.53222706495, 5891756.946288286), 61: (372629.77962331753, 5892822.561606356), 62: (373756.74884096754, 5892079.358308205)}
    plot_wind_farm_layout(example_layout, boundary_utm, title="Test Wind Farm Layout")