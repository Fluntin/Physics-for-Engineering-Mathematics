import numpy as np
import matplotlib.pyplot as plt

# Constants and initial values
wavelength_mm = 0.0005  # Wavelength in mm (5000 Ångström), approx for a green laser pointer
wave_number = 2 * np.pi / wavelength_mm  # Wave number calculation

# Geometry parameters
distance_to_screen_mm = 2000  # Distance to the detector screen in mm
original_screen_width_mm = 500 * 7.5  # Original width of the detector screen in mm
num_screen_pixels = 1000  # Number of pixels on the detector screen

# Initial slit width for different N (number of sources)
slit_width_results = np.zeros(4)
slit_width_results[0] = 20
number_of_sources_options = [2, 3, 5, 10]

# Function to calculate and plot intensity distribution
def plot_intensity_distribution(screen_width_mm, label_suffix=''):
    for index in range(1, 4):
        number_of_sources = number_of_sources_options[index]
        slit_separation_mm = 0.05 / (number_of_sources - 1)  # Separation between slits

        # Generate source positions
        source_positions = np.linspace(-slit_separation_mm / 2, slit_separation_mm / 2, number_of_sources)

        # Prepare arrays for screen positions and intensities
        screen_positions = np.linspace(-screen_width_mm / 2, screen_width_mm / 2, num_screen_pixels)
        intensity_at_screen = np.zeros(num_screen_pixels)

        # Calculate intensity distribution on the screen
        for pixel_index in range(num_screen_pixels):
            x_position = screen_positions[pixel_index]
            distance_to_pixel = np.sqrt(distance_to_screen_mm**2 + x_position**2)

            # Compute distances from all slits to current pixel
            distances_from_slits = np.sqrt(distance_to_screen_mm**2 + (x_position - source_positions)**2)

            # Intensity calculation based on superposition principle
            intensity_sum = np.float64(number_of_sources / 2)
            for i in range(number_of_sources - 1):
                for j in range(i + 1, number_of_sources):
                    intensity_sum += np.cos(wave_number * (distances_from_slits[i] - distances_from_slits[j]))

            intensity_at_screen[pixel_index] = intensity_sum / distance_to_pixel**2

        # Analyze intensity to find the width of intensity peaks
        min_intensity_positions = np.zeros([100, 2])
        valid_min_count = 1
        for i in range(1, num_screen_pixels - 1):
            if intensity_at_screen[i] <= intensity_at_screen[i - 1] and intensity_at_screen[i] <= intensity_at_screen[i + 1]:
                min_intensity_positions[valid_min_count] = [screen_positions[i], intensity_at_screen[i]]
                valid_min_count += 1

        # Compute the width of the central peak from the intensity minimum positions
        for pos_index in range(1, 99):
            if min_intensity_positions[pos_index + 1, 0] > 0:
                slit_width_results[index] = abs(min_intensity_positions[pos_index, 0] * 2)
                break

        # Plotting the intensity distribution for the current number of sources
        plt.figure(figsize=(10, 6))  # Larger figure size for better visibility
        plt.plot(screen_positions, intensity_at_screen, label=f'N = {number_of_sources} {label_suffix}', linewidth=2)
        plt.xlabel('Screen position (mm)', fontsize=14)  # Larger font size
        plt.ylabel('Intensity', fontsize=14)
        plt.title(f"Intensity Distribution for N = {number_of_sources} {label_suffix}", fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)  # Larger tick labels
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Custom grid style
        plt.legend()
        plt.show()

# Plot intensity distributions for original, 5 times, and 10 times wider screen widths
plot_intensity_distribution(original_screen_width_mm, '(Original Width)')
plot_intensity_distribution(original_screen_width_mm * 5, '(5x Width)')
plot_intensity_distribution(original_screen_width_mm * 10, '(10x Width)')

# Plotting the relationship between the number of sources and the peak width
plt.figure(figsize=(10, 6))
plt.scatter(number_of_sources_options, slit_width_results, color='red', s=100, edgecolor='black', alpha=0.75)  # Red markers with black edges
plt.xlabel('Number of sources (N)', fontsize=14)
plt.ylabel('Intensity peak width (mm)', fontsize=14)
plt.title("Intensity Peak Width vs Number of Sources", fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()

print(slit_width_results)
