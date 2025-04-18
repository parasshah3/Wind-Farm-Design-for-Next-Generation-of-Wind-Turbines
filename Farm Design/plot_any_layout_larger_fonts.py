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
    plt.scatter(coords[:, 0], coords[:, 1], s=25, label="Turbines")

    # Annotate turbines
    if annotate:
        for tid, (x, y) in zip(ids, coords):
            plt.text(x, y + 70, str(tid), fontsize=13, ha='center', color='black', fontname="Arial")

    # Axes limits with padding
    all_x = np.concatenate([boundary_coords[:, 0], coords[:, 0]])
    all_y = np.concatenate([boundary_coords[:, 1], coords[:, 1]])
    x_pad, y_pad = 1000, 1000
    plt.xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    plt.ylim(all_y.min() - y_pad, all_y.max() + y_pad)

    # Labels, grid, and formatting
    plt.xlabel("Easting (m)", fontsize=16, fontname="Arial")
    plt.ylabel("Northing (m)", fontsize=16, fontname="Arial")
    #plt.title(title, fontsize=16, fontweight='bold', fontname="Arial")
    plt.legend(prop={"family": "Arial", "size": 16})
    plt.xticks(fontsize=14, fontname="Arial")
    plt.yticks(fontsize=14, fontname="Arial")
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
    example_layout = {1: (np.float64(373690.373), np.float64(5887454.043)), 2: (np.float64(373357.37185308576), np.float64(5888554.12393752)), 3: (np.float64(373024.37070617155), np.float64(5889654.204875041)), 4: (np.float64(372691.3695592573), np.float64(5890754.285812561)), 5: (np.float64(372358.368412343), np.float64(5891854.366750081)), 6: (np.float64(372025.36726542877), np.float64(5892954.447687602)), 8: (np.float64(375755.13419223664), np.float64(5886698.356037493)), 9: (np.float64(375422.1330453224), np.float64(5887798.436975013)), 10: (np.float64(375089.1318984082), np.float64(5888898.517912534)), 11: (np.float64(374756.1307514939), np.float64(5889998.598850055)), 12: (np.float64(374423.12960457965), np.float64(5891098.679787574)), 13: (np.float64(374090.1284576654), np.float64(5892198.760725095)), 15: (np.float64(377819.8953844732), np.float64(5885942.669074987)), 16: (np.float64(377486.89423755894), np.float64(5887042.750012508)), 17: (np.float64(377153.89309064473), np.float64(5888142.830950028)), 18: (np.float64(376820.8919437305), np.float64(5889242.911887549)), 19: (np.float64(376487.8907968162), np.float64(5890342.9928250685)), 20: (np.float64(376154.88964990195), np.float64(5891443.073762589)), 22: (np.float64(379884.6565767098), np.float64(5885186.98211248)), 23: (np.float64(379551.65542979556), np.float64(5886287.063050001)), 24: (np.float64(379218.65428288135), np.float64(5887387.1439875215)), 25: (np.float64(378885.6531359671), np.float64(5888487.224925042)), 26: (np.float64(378552.6519890528), np.float64(5889587.305862562))}
    plot_wind_farm_layout(example_layout, boundary_utm, title="Test Wind Farm Layout")