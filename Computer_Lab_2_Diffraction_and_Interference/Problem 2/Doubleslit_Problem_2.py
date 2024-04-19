import numpy as np
import matplotlib.pyplot as plt
import statistics

# Constants
wavelength_mm = 0.0005 # mm = 5000 Ångström, approx wavelength for a green laser pointer
wavenumber = 2 * np.pi / wavelength_mm

# Geometry parameters
slit_separation = 0.05 # separation between slits in meters
screen_distance = 200 # distance to the detector screen in meters
screen_width = 200 # width of the detector screen in meters
number_of_points = 1000 # number of pixels on the detector screen

# Positions of the two slits relative to the center
position_slit1 = slit_separation / 2
position_slit2 = -slit_separation / 2

# Calculate the time-averaged intensity on the detector screen
screen_positions = np.empty(number_of_points)
intensities = np.empty(number_of_points)

for i in range(number_of_points):
    pixel_position = screen_width * (i / (number_of_points - 1) - 0.5)
    screen_positions[i] = pixel_position
    distance_to_pixel = np.sqrt(screen_distance**2 + pixel_position**2)
    distance_slit1_to_pixel = np.sqrt(screen_distance**2 + (pixel_position - position_slit1)**2)
    distance_slit2_to_pixel = np.sqrt(screen_distance**2 + (pixel_position - position_slit2)**2)
    intensities[i] = (1 + np.cos(wavenumber * (distance_slit1_to_pixel - distance_slit2_to_pixel))) / distance_to_pixel**2

# Finding local maxima in intensity
intensity_maxima_positions = np.zeros([100, 2])
maxima_count = 0
peak_positions = []

for i in range(len(intensities)):
    if i == number_of_points - 1:
        break
    # Check if current point is a local maximum
    if i >= 2 and (intensities[i] >= intensities[i-1] and intensities[i] >= intensities[i+1]):
        intensity_maxima_positions[maxima_count] = [screen_positions[i], intensities[i]]
        maxima_count += 1

# Calculate the distance between consecutive maxima
for i in range(maxima_count):
    if i == maxima_count - 1:
        break
    peak_positions.append(intensity_maxima_positions[i + 1, 0] - intensity_maxima_positions[i, 0])

print("dx =", statistics.median(peak_positions))

import matplotlib.pyplot as plt

# Enhanced Plot
plt.style.use('seaborn-darkgrid')  # Consistent style with previous plots
plt.figure(figsize=(10, 6))  # Set the figure size for better detail
plt.scatter(intensity_maxima_positions[:maxima_count, 0], intensity_maxima_positions[:maxima_count, 1], color='red', marker='o', label='Maxima Positions')  # Red color for maxima points
plt.plot(screen_positions, intensities, color='blue', linewidth=2, linestyle='-', label='Intensity Curve')  # Blue color for intensity curve
plt.xlabel('Position (m)', fontsize=14)  # Label for x-axis
plt.ylabel('Intensity', fontsize=14)  # Label for y-axis
plt.title('Intensity Distribution on Detector Screen', fontsize=16)  # Title for the plot
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Grid lines
plt.legend()  # Legend to distinguish plot elements
plt.show()

