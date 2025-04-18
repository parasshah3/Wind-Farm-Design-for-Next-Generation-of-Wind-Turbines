import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import time

from farm_design_functions_v27 import evaluate_all_arrangements

# ---- INPUTS ----

file_path = "/Users/paras/Desktop/3YP Python Scripts/Farm Design/Wind Farms .csv Files/25MW_EighthIteration.xlsm"
sheet_name = "WindDist"

boundary_utm = np.array([
    (371526.684, 5893424.206),
    (378450.513, 5890464.698),
    (380624.287, 5884484.707),
    (373690.373, 5887454.043),
    (371526.684, 5893424.206)  # close loop
])

Ct = 0.790
r_0 = 124.7
z = 140
z_0 = 7.8662884488e-04

wind_rose = {
    0: 0.06, 30: 0.03, 60: 0.04, 90: 0.06, 120: 0.05, 150: 0.07,
    180: 0.09, 210: 0.17, 240: 0.19, 270: 0.12, 300: 0.07, 330: 0.05
}
primary_direction = 240

A = 7529.81
B = 6350.15
alpha_deg = 133.36

a_steps = 15
b_steps = 15
theta_steps = 7

# ---- RUN OPTIMISATION ----

start_time = time.time()

sorted_results, best_layout, best_LCOE, best_a, best_b, best_theta_deg = evaluate_all_arrangements(
    file_path, sheet_name, boundary_utm,
    Ct, r_0, z, z_0, wind_rose, A, B, alpha_deg,
    a_steps, b_steps, theta_steps
)

end_time = time.time()
elapsed_time = end_time - start_time

# ---- PRINT RESULTS ----
print(f"Total number of layouts evaluated: {len(sorted_results)}")
print(f"\n✅ Finished in {elapsed_time:.4f} seconds\n")
print(f"finished in {elapsed_time/60:.4f} minutes\n")
print(f"finished in {elapsed_time/3600:.4f} hours\n")


print("Top 5 Regular Layouts:")
for idx, (layout, lcoe, a, b, theta) in enumerate(sorted_results[:5]):
    print(f"Rank {idx+1}: LCOE = {lcoe:.3f} £/MWhr | Turbines = {len(layout)} | a = {a:.1f} m, b = {b:.1f} m, θ = {theta:.1f}°")


print(f"Best LCOE: {best_LCOE:.3f} £/MWhr")
print(f"Best a: {best_a}")
print(f"Best b: {best_b}")
print(f"Best theta_deg: {best_theta_deg}")
print(f"Best Regular Layout: {best_layout}")

# ---- PLOT BEST LAYOUT ----

plt.figure(figsize=(8, 8))
plt.plot(boundary_utm[:, 0], boundary_utm[:, 1], 'k-', label="Boundary")
turbine_coords = np.array(list(best_layout.values()))
plt.scatter(turbine_coords[:, 0], turbine_coords[:, 1], c='red', label=f"Turbines ({len(best_layout)})", s=15)

for tid, (x, y) in best_layout.items():
    plt.text(x, y + 20, str(tid), fontsize=6, ha='center')

plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.title(f"Best Regular Layout\nLCOE = {best_LCOE:.2f} £/MWhr")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---- PLOT LCOE vs. Layout Index ----

lcoe_values = [lcoe for _, lcoe, _, _, _ in sorted_results]

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(lcoe_values) + 1), lcoe_values, marker='o', linestyle='-')
plt.xlabel("Layout Index (Sorted by LCOE)")
plt.ylabel("LCOE (Lifetime) [£/MWhr]")
plt.title("LCOE vs. Layout Index")
plt.grid(True)
plt.tight_layout()
plt.show()