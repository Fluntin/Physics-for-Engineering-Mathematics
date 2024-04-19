import numpy as np
import matplotlib.pyplot as plt

# Wave parameters
wavelength_mm = 0.0005  # mm = 500 nanometers, typical wavelength for green laser pointer
wave_number = 2 * np.pi / wavelength_mm

# Geometry parameters
distance_between_slits_mm = 0.5  # distance between the two slits in mm
distance_to_screen_mm = 2000  # distance from slits to the detector screen in mm
screen_width_mm = 100  # width of the detector screen in mm
num_screen_points = 1000  # number of points (pixels) on the detector screen

# Single slit parameters
num_sources_per_slit = 10  # number of point sources per slit
slit_width_mm = 0.05  # width of each slit in mm

############ Double Slit ##############

# Calculate source positions for a single slit
single_slit_sources = np.linspace(-slit_width_mm / 2, slit_width_mm / 2, int(num_sources_per_slit / 2))

# Calculate source positions for double slit setup
double_slit_source1 = single_slit_sources + distance_between_slits_mm / 2
double_slit_source2 = single_slit_sources - distance_between_slits_mm / 2

all_sources = np.concatenate((double_slit_source1, double_slit_source2), axis=None)

# Compute time-averaged intensity on the detector screen for double slit
screen_positions = np.empty(num_screen_points)
intensity_pattern_double = np.empty(num_screen_points)

for index in range(num_screen_points):
    position_on_screen = screen_width_mm * (index / (num_screen_points - 1) - 0.5)
    screen_positions[index] = position_on_screen
    distance_to_screen = np.sqrt(distance_to_screen_mm ** 2 + position_on_screen ** 2)

    distances_to_sources = np.sqrt(distance_to_screen_mm ** 2 + (position_on_screen - all_sources) ** 2)
    double_slit_intensity_sum = np.float64(num_sources_per_slit / 2)
    for l in range(num_sources_per_slit - 1):
        for m in range(l + 1, num_sources_per_slit):
            double_slit_intensity_sum += np.cos(wave_number * (distances_to_sources[l] - distances_to_sources[m]))

    intensity_pattern_double[index] = double_slit_intensity_sum / distance_to_screen ** 2

# Normalize the intensity patterns to have the same central peak amplitude
intensity_pattern_double /= np.max(intensity_pattern_double)

############ Single Slit ##############

# Compute time-averaged intensity on the detector screen for single slit
source_positions_single = np.linspace(-slit_width_mm / 2, slit_width_mm / 2, num_sources_per_slit)
intensity_pattern_single = np.empty(num_screen_points)

for index in range(num_screen_points):
    position_on_screen = screen_width_mm * (index / (num_screen_points - 1) - 0.5)
    screen_positions[index] = position_on_screen
    distance_to_screen = np.sqrt(distance_to_screen_mm ** 2 + position_on_screen ** 2)

    distances_to_sources = np.sqrt(distance_to_screen_mm ** 2 + (position_on_screen - source_positions_single) ** 2)
    single_slit_intensity_sum = np.float64(num_sources_per_slit / 2)
    for l in range(num_sources_per_slit - 1):
        for m in range(l + 1, num_sources_per_slit):
            single_slit_intensity_sum += np.cos(wave_number * (distances_to_sources[l] - distances_to_sources[m]))

    intensity_pattern_single[index] = single_slit_intensity_sum / distance_to_screen ** 2

intensity_pattern_single /= np.max(intensity_pattern_single)

# Plot the results
plt.style.use('seaborn-darkgrid')  # Consistent style with previous plots
plt.figure(figsize=(10, 6))  # Set the figure size for better detail
plt.plot(screen_positions, intensity_pattern_single, label='Single Slit')
plt.plot(screen_positions, intensity_pattern_double, label='Double Slit')
plt.xlabel('Position on Screen (mm)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
