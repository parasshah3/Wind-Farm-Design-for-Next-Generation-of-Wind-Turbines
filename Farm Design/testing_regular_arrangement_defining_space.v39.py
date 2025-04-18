import numpy as np
import matplotlib.pyplot as plt
from farm_design_functions_v25 import define_parameter_space

# Inputs
rotor_diameter = 249.4  # e.g. 2 * r_0
A = 7529.81  # from earlier length calculation
B = 6350.15
alpha_deg = 133.36

# Call the function
param_space = define_parameter_space(rotor_diameter, A, B, alpha_deg)
print(f"Generated {len(param_space)} parameter combinations.")
print(f"Example parameter combination: {param_space[0]}")
print(f"Example parameter combination: {param_space[100]}")