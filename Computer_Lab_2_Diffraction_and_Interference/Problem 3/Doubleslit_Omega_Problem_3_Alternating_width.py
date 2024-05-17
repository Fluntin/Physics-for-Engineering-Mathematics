import numpy as np
import matplotlib.pyplot as plt

# Constants for wavelength and wave number
wavelength_mm = 0.005  # mm = 5000 Ångström, approx wavelength for a green laser pointer
wave_number = 2 * np.pi / wavelength_mm

# Detector screen geometry parameters
distance_to_screen_mm = 2000 * 5  # distance to detector screen in mm
original_screen_width_mm = 100  # original width of detector screen in mm
number_of_points = 5000  # number of points (pixels) on the detector screen
number_of_slits_options = [2, 3, 5, 10]  # different number of slits to be evaluated
omega = 0.005  # constant slit separation multiplier

# Initial peak widths array to store results for different N
peak_widths = np.zeros(len(number_of_slits_options))

# Function to calculate and plot intensity distribution
def plot_intensity_distribution(screen_width_mm, label_suffix=''):
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn-darkgrid')

    for n_index, number_of_slits in enumerate(number_of_slits_options):
        slit_separation_mm = omega / (number_of_slits - 1) if number_of_slits > 1 else 0

        # Generate slit positions
        slit_positions = np.linspace(-slit_separation_mm / 2, slit_separation_mm / 2, number_of_slits)

        # Calculate time-averaged intensity on the detector screen
        screen_positions = np.linspace(-screen_width_mm / 2, screen_width_mm / 2, number_of_points)
        intensities = np.zeros(number_of_points)

        for point_index in range(number_of_points):
            position_on_screen_mm = screen_positions[point_index]
            distance_from_screen_mm = np.sqrt(distance_to_screen_mm ** 2 + position_on_screen_mm ** 2)

            distance_from_slits = np.sqrt(distance_to_screen_mm ** 2 + (position_on_screen_mm - slit_positions) ** 2)

            intensity_sum = np.float64(number_of_slits / 2)
            for first_slit in range(number_of_slits - 1):
                for second_slit in range(first_slit + 1, number_of_slits):
                    intensity_sum += np.cos(wave_number * (distance_from_slits[first_slit] - distance_from_slits[second_slit]))

            intensities[point_index] = intensity_sum / distance_from_screen_mm ** 2

        min_positions = np.zeros([1000, 2])
        count = 1
        for intensity_index in range(1, len(intensities) - 1):
            if intensities[intensity_index] <= intensities[intensity_index - 1] and intensities[intensity_index] <= intensities[intensity_index + 1]:
                min_positions[count] = [screen_positions[intensity_index], intensities[intensity_index]]
                count += 1

        for pos_index in range(1, len(min_positions) - 1):
            if min_positions[pos_index + 1, 0] > 0:
                peak_widths[n_index] = abs(min_positions[pos_index, 0] * 2)
                break

        # Plot the intensity distribution for the current number of slits
        plt.plot(screen_positions, intensities, label=f'N = {number_of_slits} {label_suffix}')

    # Enhance and finalize the plot for intensity distributions
    plt.xlabel('Position on screen (mm)', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.title(f'Intensity Distribution for Varying Number of Slits {label_suffix}', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

# Plot intensity distributions for original, 5 times, and 10 times wider screen widths
plot_intensity_distribution(original_screen_width_mm, '(Original Width)')
plot_intensity_distribution(original_screen_width_mm * 5, '(5x Width)')
plot_intensity_distribution(original_screen_width_mm * 10, '(10x Width)')

print(peak_widths)
