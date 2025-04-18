import numpy as np
from scipy import special

# Define the equation for finding the shape factor k
def equation(k, sigma, U_mean):
    term1 = special.gamma(1 + 2 / k) / (special.gamma(1 + 1 / k) ** 2)
    return (sigma**2 / U_mean**2) - term1 + 1

# Derivative of the equation w.r.t. k
def equation_derivative(k, sigma, U_mean):
    term1 = special.gamma(1 + 2 / k) / (special.gamma(1 + 1 / k) ** 2)
    gamma_1 = special.gamma(1 + 1 / k)
    gamma_2 = special.gamma(1 + 2 / k)
    term1 = gamma_2 / (gamma_1 ** 2)
    
    # Derivative of the gamma terms
    d_gamma_1 = -special.psi(1 + 1 / k) * gamma_1 / (k**2)
    d_gamma_2 = -special.psi(1 + 2 / k) * gamma_2 / (k**2)
    
    d_term1 = (d_gamma_2 / (gamma_1**2)) - (2 * gamma_2 * d_gamma_1 / (gamma_1**3))
    return -d_term1

# Newton-Raphson method to find k
def newton_raphson(U_mean, sigma, k_initial=2.0, tol=1e-6, max_iter=100):
    k = k_initial
    for i in range(max_iter):
        f_val = equation(k, sigma, U_mean)
        f_prime = equation_derivative(k, sigma, U_mean)
        
        if f_prime == 0:
            raise ValueError("Zero derivative encountered. Newton-Raphson method failed.")
        
        # Update k
        k_new = k - f_val / f_prime
        print(k_new)
        # Check for convergence
        if abs(k_new - k) < tol:
            return k_new
            
        
        k = k_new
    
    raise ValueError("Newton-Raphson method did not converge within the maximum iterations.")

# Input values
U_mean = 9.075548172  # Mean wind speed (m/s)
sigma = 2.318815231  # Standard deviation (m/s)

# Solve for shape factor k using Newton-Raphson method
try:
    k_newton = newton_raphson(U_mean, sigma, k_initial=2.0)
    c_newton = U_mean / special.gamma(1 + 1 / k_newton)
    print(f"Newton-Raphson Method - Shape Factor (k): {k_newton}")
    print(f"Newton-Raphson Method - Scale Factor (c): {c_newton}")
except ValueError as e:
    print(f"Error: {e}")