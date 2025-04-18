import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})

# Load wind turbine UTM positions from CSV
file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Turbine positions UTM.csv"
df = pd.read_csv(file_path, header=None, names=["Easting", "Northing"])

# Store wind turbine positions in a dictionary
turbine_positions = {i+1: (row["Easting"], row["Northing"]) for i, row in df.iterrows()}

# Define boundary UTM coordinates
boundary_utm = [
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # Closing the boundary loop
]

# Wind rose data: Wind direction (degrees) and relative wind speed (scaling factor)
wind_rose = {
    0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
    180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
}

# Extract X and Y coordinates for plotting
turbine_x, turbine_y = zip(*turbine_positions.values())
boundary_x, boundary_y = zip(*boundary_utm)

# Determine bottom-left corner for wind rose
min_x, min_y = min(turbine_x), min(turbine_y)
wind_rose_x, wind_rose_y = min_x - 200, min_y - 200  # Offset to position the wind rose

#Create the plot
plt.figure(figsize=(8, 8))
plt.scatter(turbine_x, turbine_y, marker='o',s=15, label="Wind Turbines")
plt.plot(boundary_x, boundary_y, linestyle='-', color='black', label="Wind Farm Boundary", alpha=0.7, linewidth =0.85)

# Annotate turbine numbers
for turbine_id, (x, y) in turbine_positions.items():
    plt.text(x, y + 70, str(turbine_id), fontsize=10, ha="center", color="black", fontname="Arial")  # Keep slightly smaller

# Plot wind speed rose in bottom-left corner
for direction, speed in wind_rose.items():
    angle_rad = np.radians(270 - direction)
    dx = speed * 10000 * np.cos(angle_rad)
    dy = speed * 10000 * np.sin(angle_rad)

    plt.arrow(wind_rose_x, wind_rose_y, dx, dy, 
              head_width=200, head_length=300, fc='green', ec='green', alpha=0.6)

# Labels and formatting
plt.xlabel("Easting (m)", fontsize=14, fontname="Arial")
plt.ylabel("Northing (m)", fontsize=14, fontname="Arial")
#plt.title("Sheringham Shoal Wind Farm - Turbine Positions and Wind Rose", fontsize=16, fontweight='bold', fontname="Arial")
plt.legend(prop={"family": "Arial", "size": 14})
plt.xticks(fontsize=12, fontname="Arial")
plt.yticks(fontsize=12, fontname="Arial")
plt.grid(True, linestyle='-', linewidth=0.7, alpha=0.7)

import matplotlib.ticker as ticker

ax = plt.gca()
ax.ticklabel_format(style='scientific', axis='both', scilimits=(-3, 4), useMathText=True)

# Change offset text size (the ×10⁶ label in the corner)
ax.xaxis.offsetText.set_fontsize(10)
ax.yaxis.offsetText.set_fontsize(10)

# Change offset text font
ax.xaxis.offsetText.set_fontname("Arial")
ax.yaxis.offsetText.set_fontname("Arial")

plt.tight_layout()
plt.show()