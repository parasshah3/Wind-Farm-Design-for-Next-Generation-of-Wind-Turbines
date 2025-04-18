import sys
import numpy as np
from scipy import special
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QLabel, QLineEdit, QPushButton, QWidget, QGroupBox, QGridLayout

# Predefined Data: Heights, Mean Wind Speeds, and Standard Deviations
heights_data = np.array([50, 75, 100, 150, 200, 250, 500])  # Heights (z)
mean_wind_speeds = np.array([8.556808472, 8.854023933, 9.075548172, 9.396860123, 9.631995201, 9.813015938, 10.31694698])
std_devs = np.array([2.168092489, 2.253010273, 2.318815231, 2.423012495, 2.50705266, 2.573358536, 2.759818316])

# Perform Linear Regression for Mean Wind Speed and Standard Deviation
log_heights = np.log(heights_data)

def perform_regression(y_values):
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(log_heights, y_values)
    return slope, intercept

# Regression for mean wind speed
A_mean, intercept_mean = perform_regression(mean_wind_speeds)
z0_mean = np.exp(-intercept_mean / A_mean)

# Regression for standard deviation
B_std, intercept_std = perform_regression(std_devs)
z0_std = np.exp(-intercept_std / B_std)

# Define relationships for mean wind speed and standard deviation
def calculate_mean_wind_speed(height):
    return A_mean * np.log(height / z0_mean)

def calculate_std_dev(height):
    return B_std * np.log(height / z0_std)

# Weibull root-finding equation
def equation(k, sigma, U_mean):
    term1 = special.gamma(1 + 2 / k) / (special.gamma(1 + 1 / k) ** 2)
    return (sigma**2 / U_mean**2) - term1 + 1

# Weibull distribution function
def weibull_distribution(U, k, c):
    return (k / c) * (U / c) ** (k - 1) * np.exp(-(U / c) ** k)

# Calculate Weibull parameters
def calculate_weibull_params(height):
    U_mean = calculate_mean_wind_speed(height)
    sigma = calculate_std_dev(height)
    k = brentq(equation, 0.5, 10, args=(sigma, U_mean))
    c = U_mean / special.gamma(1 + 1 / k)
    return k, c, U_mean, sigma

# PyQt5 GUI
class WeibullApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Weibull Distribution and Logarithmic Relationships")
        self.layout = QVBoxLayout()

        # Input Section
        input_group = QGroupBox("Input Parameters")
        input_layout = QGridLayout()

        input_layout.addWidget(QLabel("Enter Height Above Sea Level (m):"), 0, 0)
        self.height_input = QLineEdit()
        input_layout.addWidget(self.height_input, 0, 1)

        input_layout.addWidget(QLabel("Wind Speed for Probability (m/s):"), 1, 0)
        self.specific_speed_input = QLineEdit()
        input_layout.addWidget(self.specific_speed_input, 1, 1)

        input_group.setLayout(input_layout)
        self.layout.addWidget(input_group)

        # Buttons
        button_layout = QVBoxLayout()
        self.plot_button = QPushButton("Plot Weibull Distribution")
        self.plot_button.clicked.connect(self.plot_weibull)
        button_layout.addWidget(self.plot_button)

        self.probability_button = QPushButton("Calculate Probability")
        self.probability_button.clicked.connect(self.calculate_probability)
        button_layout.addWidget(self.probability_button)

        self.layout.addLayout(button_layout)

        # Output Section
        self.output_label = QLabel("Output will appear here.")
        self.layout.addWidget(self.output_label)

        self.setLayout(self.layout)

    def plot_weibull(self):
        try:
            height = float(self.height_input.text())
            k, c, U_mean, sigma = calculate_weibull_params(height)

            # Generate Weibull distribution
            U_values = np.linspace(0, 2 * U_mean, 500)
            weibull_pdf = weibull_distribution(U_values, k, c)

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(U_values, weibull_pdf, color='red', linewidth=1,
                     label=f"Weibull Distribution\nHeight: {height:.1f}m\nk={k:.3f}, c={c:.3f}")
            plt.xlabel("Wind Speed (m/s)")
            plt.ylabel("Probability Density (1/m/s)")
            plt.title("Weibull Distribution for Specified Height")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            self.output_label.setText(f"Height: {height:.1f}m\nMean Wind Speed: {U_mean:.3f} m/s\n"
                                      f"Standard Deviation: {sigma:.3f} m/s\nShape Factor (k): {k:.3f}\n"
                                      f"Scale Factor (c): {c:.3f}")
        except Exception as e:
            self.output_label.setText(f"Error: {e}")

    def calculate_probability(self):
        try:
            height = float(self.height_input.text())
            specific_speed = float(self.specific_speed_input.text())

            k, c, _, _ = calculate_weibull_params(height)
            prob_speed = weibull_distribution(specific_speed, k, c)

            self.output_label.setText(f"Probability of Wind Speed = {specific_speed} m/s\n"
                                      f"Shape Factor (k): {k:.3f}, Scale Factor (c): {c:.3f}\n"
                                      f"Probability: {prob_speed:.5f}")
        except Exception as e:
            self.output_label.setText(f"Error: {e}")

# Run the application
app = QApplication(sys.argv)
window = WeibullApp()
window.show()
sys.exit(app.exec_())