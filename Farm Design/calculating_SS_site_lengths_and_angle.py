import numpy as np
import math

# Sheringham Shoal UTM boundary coordinates
boundary_utm = np.array([
    (371526.684, 5893424.206),  # P1
    (378450.513, 5890464.698),  # P2
    (380624.287, 5884484.707),  # P3
    (373690.373, 5887454.043),  # P4
    (371526.684, 5893424.206)   # P1 again to close loop
])

# Label key vertices
P1 = boundary_utm[0]
P2 = boundary_utm[1]
P3 = boundary_utm[2]
P4 = boundary_utm[3]  # use P4 not P3, since P4 shares vertex with P1

# Compute side lengths
A = np.linalg.norm(P2 - P1)  # length along dominant direction
B = np.linalg.norm(P4 - P1)  # length orthogonal to A

pthree_to_pone = np.linalg.norm(P3 - P1)  # length from P3 to P1

# Using cosine rule to find alpha
alpha_rad = np.acos((A**2 + B**2 - pthree_to_pone**2) / (2 * A * B))

alpha_deg = math.degrees(alpha_rad)

# Output results
print(f"Side A length: {A:.5f} m")
print(f"Side B length: {B:.5f} m")
print(f"Boundary angle α: {alpha_deg:.5f}°")