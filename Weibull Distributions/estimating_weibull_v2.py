import numpy as np

# Wind speed data and percentiles
wind_speeds = np.array([9.79, 9.79, 9.79, 9.79, 9.79, 9.79, 9.79, 9.78, 9.78, 9.78,
                        9.78, 9.78, 9.78, 9.78, 9.78, 9.78, 9.78, 9.77, 9.77, 9.77,
                        9.77, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77, 9.77,
                        9.77, 9.77, 9.77, 9.77, 9.76, 9.76, 9.76, 9.76, 9.76, 9.76,
                        9.76, 9.76, 9.76, 9.76, 9.76, 9.76, 9.76, 9.76, 9.76,9.76])
percentiles = np.array([2, 3.99, 5.99, 7.99, 9.99, 11.98, 13.98, 15.98, 17.98, 19.97,
                        21.97, 23.97, 25.97, 27.96, 29.96, 31.96, 33.95, 35.95, 37.95,
                        39.95, 41.94, 43.94, 45.94, 47.94, 50.07, 52.06, 54.06, 56.06,
                        58.06, 60.05, 62.05, 64.05, 66.05, 68.04, 70.04, 72.04, 74.03,
                        76.03, 78.03, 80.03, 82.02, 84.02, 86.02, 88.02, 90.01, 92.01,
                        94.01, 96.01, 98, 100])

#calculate length of wind_speeds and percentiles arrays
n = len(wind_speeds)
m = len(percentiles)
print(n)
print(m)

# Calculate interval weights
weights = np.diff(percentiles, prepend=0)  # Differences between consecutive percentiles
weights[-1] = 2  # Manually set the last interval

# Calculate weighted mean
weighted_mean = np.sum(wind_speeds * weights) / np.sum(weights)

# Calculate weighted variance
weighted_variance = np.sum(weights * (wind_speeds - weighted_mean)**2) / np.sum(weights)

# Take the square root to get the standard deviation
weighted_std_dev = np.sqrt(weighted_variance)

print(f"Weighted Mean Wind Speed: {weighted_mean:.3f} m/s")
print(f"Weighted Standard Deviation: {weighted_std_dev:.3f} m/s")