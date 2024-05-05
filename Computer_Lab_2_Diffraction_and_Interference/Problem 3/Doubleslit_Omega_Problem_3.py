import numpy as np
import matplotlib.pyplot as plt

# Constants for wavelength and wave number
wavelength_mm = 0.0005  # mm = 5000 Ångström, approx wavelength for a green laser pointer
wave_number = 2 * np.pi / wavelength_mm

# Detector screen geometry parameters
distance_to_screen_mm = 2000  # distance to detector screen in mm
screen_width_mm = 20  # width of detector screen in mm
number_of_points = 1000  # number of points (pixels) on the detector screen
peak_widths = np.zeros(4)  # array to store peak widths
peak_widths[0] = 20  # initial peak width set
number_of_slits = 3  # number of slits
omega_values = [0.05, 0.15, 0.25, 0.5]  # array of slit separation multipliers

for index in range(4):
    slit_separation_mm = omega_values[index] / (number_of_slits - 1)  # calculate slit separation
    slit_positions = np.zeros((number_of_slits))  # array to store slit positions
    slit_positions = np.linspace(-slit_separation_mm / 2, slit_separation_mm / 2, number_of_slits)
    print(slit_positions)

    # Calculate time averaged intensity on the detector screen
    screen_positions = np.empty(number_of_points)
    intensities = np.empty(number_of_points)
    for point_index in range(number_of_points):
        position_on_screen_mm = screen_width_mm * (point_index / (number_of_points - 1) - 0.5)
        screen_positions[point_index] = position_on_screen_mm
        distance_from_screen_mm = np.sqrt(distance_to_screen_mm ** 2 + position_on_screen_mm ** 2)

        distance_from_slits = [np.sqrt(distance_to_screen_mm ** 2 + (position_on_screen_mm - pos) ** 2) for pos in slit_positions]

        intensity_sum = np.float64(number_of_slits / 2)
        for first_slit in range(number_of_slits - 1):
            for second_slit in range(first_slit + 1, number_of_slits):
                intensity_sum += np.cos(wave_number * (distance_from_slits[first_slit] - distance_from_slits[second_slit]))

        intensities[point_index] = intensity_sum / distance_from_screen_mm ** 2

    min_positions = np.zeros([1000, 2])
    count = 1
    for intensity_index in range(len(intensities)):
        if intensity_index == 999:
            break
        if intensity_index >= 2 and (intensities[intensity_index] <= intensities[intensity_index - 1] and intensities[intensity_index] <= intensities[intensity_index + 1]):
            min_positions[count] = [screen_width_mm * (intensity_index / (number_of_points - 1) - 0.5), intensities[intensity_index]]
            count += 1

    for pos_index in range(len(min_positions)):
        if pos_index == 99:
            break
        if pos_index >= 1 and min_positions[pos_index + 1, 0] > 0:
            print("Width: ", abs(min_positions[pos_index, 0] * 2))
            peak_widths[index] = abs(min_positions[pos_index, 0] * 2)
            break

plt.style.use('seaborn-darkgrid')  # Consistent style with previous plots
plt.figure(figsize=(10, 6))  # Set the figure size for better detail
plt.plot(screen_positions, intensities)
plt.xlabel('Position on screen (mm)')
plt.ylabel('Intensity')
plt.title(f"Omega = 3")

plt.style.use('seaborn-darkgrid')  # Consistent style with previous plots
plt.figure(figsize=(10, 6))  # Set the figure size for better detail
plt.scatter(omega_values, peak_widths)
plt.xlabel('\u03C9')
plt.ylabel('Intensity peak width (mm)')
plt.title("Intensity peak width vs. width of the slit")
plt.show()

print(peak_widths)