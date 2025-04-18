import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Input data
heights = np.array([50, 75, 100, 150, 200, 250, 500])
wind_speeds = np.array([8.556808472, 8.854023933, 9.075548172, 
                        9.396860123, 9.631995201, 9.813015938, 10.31694698])
wind_std_dev = np.array([2.168092489, 2.253010273, 2.318815231, 
                         2.423012495, 2.50705266, 2.573358536, 2.759818316])

# Transform heights to natural logarithm
log_heights = np.log(heights)

# Linear regression for Mean Wind Speed
slope_U, intercept_U, r_value_U, _, _ = linregress(log_heights, wind_speeds)
z0_U = np.exp(-intercept_U / slope_U)

# Linear regression for Standard Deviation
slope_sigma, intercept_sigma, r_value_sigma, _, _ = linregress(log_heights, wind_std_dev)
alpha_sigma = np.exp(-intercept_sigma / slope_sigma)

# Generate smooth range
smooth_heights = np.linspace(10, 550, 500)
predicted_wind_speeds = slope_U * np.log(smooth_heights / z0_U)
predicted_std_dev = slope_sigma * np.log(smooth_heights / alpha_sigma)

# ---- Combined Plots ----
# ---- Combined Plots ----
import matplotlib.ticker as ticker
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Mean Wind Speeds
axes[0].scatter(heights, wind_speeds, s=30, label='Measured Mean Wind Speeds', zorder=3)
axes[0].plot(smooth_heights, predicted_wind_speeds, color='orange', linestyle='--', linewidth=1.2,
             label = fr'Fitted: $A$={slope_U:.3f}, $z_0$={z0_U:.4f} $\times 10^{{-4}}$ m')
axes[0].set_xlabel(r'Height above sea-level, $z$ (m)', fontsize=14)
axes[0].set_ylabel(r'Mean Wind Speed, $\overline{U}$ (m/s)', fontsize=14)
#axes[0].set_title('Mean Wind Speed vs Height', fontsize=16, fontweight='bold')
axes[0].legend(prop={"family": "Arial", "size": 14})
axes[0].grid(True, linestyle='-', linewidth=0.7, alpha=0.6)
axes[0].tick_params(axis='both', labelsize=12)

# Plot Standard Deviation
axes[1].scatter(heights, wind_std_dev, color='green', s=30, label='Measured Standard Deviation', zorder=3)
axes[1].plot(smooth_heights, predicted_std_dev, color='orange', linestyle='--', linewidth=1.2,
             label = fr'Fitted: $B$={slope_sigma:.3f}, $\alpha$={alpha_sigma:.4f} $\times 10^{{-2}}$ m')
axes[1].set_xlabel(r'Height above sea-level, $z$ (m)', fontsize=14)
axes[1].set_ylabel(r'Standard Deviation, $\sigma$ (m/s)', fontsize=14)
#axes[1].set_title('Wind Speed Std Dev vs Height', fontsize=16, fontweight='bold')
axes[1].legend(prop={"family": "Arial", "size": 14})
axes[1].grid(True, linestyle='-', linewidth=0.7, alpha=0.6)
axes[1].tick_params(axis='both', labelsize=12)

# Scientific axis formatting
for ax in axes:
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(-3, 4), useMathText=True)
    ax.xaxis.offsetText.set_fontsize(10)
    ax.yaxis.offsetText.set_fontsize(10)
    ax.xaxis.offsetText.set_fontname("Arial")
    ax.yaxis.offsetText.set_fontname("Arial")

plt.tight_layout()
plt.show()