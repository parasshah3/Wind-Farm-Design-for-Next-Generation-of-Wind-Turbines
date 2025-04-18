import sys
import numpy as np
from scipy import special
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget, QComboBox, QCheckBox, QGroupBox, QGridLayout

# Data for heights
data = {
    50: {"mean": 8.556808472, "std_dev": 2.168092489},
    75: {"mean": 8.854023933, "std_dev": 2.253010273},
    100: {"mean": 9.075548172, "std_dev": 2.318815231},
    150: {"mean": 9.396860123, "std_dev": 2.423012495},
    200: {"mean": 9.631995201, "std_dev": 2.50705266},
    250: {"mean": 9.813015938, "std_dev": 2.573358536},
    500: {"mean": 10.31694698, "std_dev": 2.759818316},
}

# Define the root-finding equation for the shape factor k
def equation(k, sigma, U_mean):
    term1 = special.gamma(1 + 2 / k) / (special.gamma(1 + 1 / k) ** 2)
    return (sigma**2 / U_mean**2) - term1 + 1

# Weibull distribution PDF
def weibull_distribution(U, k, c):
    return (k / c) * (U / c) ** (k - 1) * np.exp(-(U / c) ** k)

# Calculate Weibull parameters
def calculate_weibull_params(height):
    U_mean = data[height]["mean"]
    sigma = data[height]["std_dev"]
    k = brentq(equation, 0.5, 10, args=(sigma, U_mean))
    c = U_mean / special.gamma(1 + 1 / k)
    return k, c

# PyQt5 GUI
class WeibullApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Weibull Distribution and Altitude Analysis")
        self.layout = QVBoxLayout()

        # Input Section
        input_group = QGroupBox("Input Parameters")
        input_layout = QGridLayout()

        # Height selection
        input_layout.addWidget(QLabel("Select Height Above Sea Level (m):"), 0, 0)
        self.height_select = QComboBox()
        self.height_select.addItems([str(h) for h in data.keys()])
        input_layout.addWidget(self.height_select, 0, 1)

        # Specific speed input for probability
        input_layout.addWidget(QLabel("Wind Speed for Probability (m/s):"), 1, 0)
        self.specific_speed_input = QLineEdit()
        input_layout.addWidget(self.specific_speed_input, 1, 1)

        # Plot all checkbox
        self.plot_all_checkbox = QCheckBox("Plot Weibull Distributions for All Heights")
        input_layout.addWidget(self.plot_all_checkbox, 2, 0, 1, 2)

        input_group.setLayout(input_layout)
        self.layout.addWidget(input_group)

        # Output Section
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()

        self.output_label = QLabel("Output will appear here.")
        output_layout.addWidget(self.output_label)

        # Buttons
        self.plot_button = QPushButton("Plot Weibull Distribution")
        self.plot_button.clicked.connect(self.plot_weibull)
        output_layout.addWidget(self.plot_button)

        self.calculate_button = QPushButton("Calculate Probability")
        self.calculate_button.clicked.connect(self.calculate_probability)
        output_layout.addWidget(self.calculate_button)

        self.plot_error_button = QPushButton("Plot Error Bars of Mean Wind Speed")
        self.plot_error_button.clicked.connect(self.plot_error_bars)
        output_layout.addWidget(self.plot_error_button)

        output_group.setLayout(output_layout)
        self.layout.addWidget(output_group)

        self.setLayout(self.layout)

    def plot_weibull(self):
        try:
            height = int(self.height_select.currentText())
            plot_all = self.plot_all_checkbox.isChecked()

            plt.rcParams.update({"font.family": "Arial", "font.size": 14})
            fig, ax = plt.subplots(figsize=(8, 5))

            if plot_all:
                for h, stats in data.items():
                    k, c = calculate_weibull_params(h)
                    U_vals = np.linspace(0, 2 * stats["mean"], 500)
                    pdf = weibull_distribution(U_vals, k, c)
                    ax.plot(U_vals, pdf, label=f"{h} m: $k$={k:.2f}, $\\lambda$={c:.2f}")
            else:
                k, c = calculate_weibull_params(height)
                U_vals = np.linspace(0, 2 * data[height]["mean"], 500)
                pdf = weibull_distribution(U_vals, k, c)
                ax.plot(U_vals, pdf, label=f"{height} m: $k$={k:.2f}, $\\lambda$={c:.2f}")

            ax.set_xlabel(r"Wind Speed, $U$ (m/s)")
            ax.set_ylabel(r"Probability Density, $p(U)$ (1/(m/s))")
            ax.grid(True, linestyle="-", linewidth=0.7, alpha=0.7)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(prop={"family": "Arial", "size": 14})
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.output_label.setText(f"Error: {e}")

    def calculate_probability(self):
        try:
            height = int(self.height_select.currentText())
            specific_speed = float(self.specific_speed_input.text())

            k, c = calculate_weibull_params(height)
            prob_speed = weibull_distribution(specific_speed, k, c)

            self.output_label.setText(
                f"Height: {height}m\n"
                f"Shape Factor (k): {k:.10f}\n"
                f"Scale Factor (c): {c:.10f}\n"
                f"Probability of Wind Speed = {specific_speed} m/s: {prob_speed:.5f}"
            )
        except ValueError:
            self.output_label.setText("Error: Invalid input. Please enter numerical values.")
        except Exception as e:
            self.output_label.setText(f"Error: {e}")

    def plot_error_bars(self):
        heights = list(data.keys())
        means = [data[h]["mean"] for h in heights]
        stds = [data[h]["std_dev"] for h in heights]

        plt.rcParams.update({"font.family": "Arial", "font.size": 14})
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.errorbar(heights, means, yerr=stds, fmt='o', capsize=5,
                    label="Mean Wind Speed ± Std Dev")

        ax.set_xlabel("Height Above Sea Level (m)")
        ax.set_ylabel("Wind Speed (m/s)")
        ax.grid(True, linestyle="-", linewidth=0.7, alpha=0.7)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(prop={"family": "Arial", "size": 14})
        plt.tight_layout()
        plt.show()

# Run the application
app = QApplication(sys.argv)
window = WeibullApp()
window.show()
sys.exit(app.exec_())