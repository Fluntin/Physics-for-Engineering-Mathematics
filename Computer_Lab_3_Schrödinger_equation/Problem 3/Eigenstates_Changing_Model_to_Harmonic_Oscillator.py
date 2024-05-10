import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------
# Constants and Parameters
#----------------------------------------
num_points = 20000
step_length = 5.0 / num_points
step_length_squared = step_length ** 2


#----------------------------------------
# Harmonic Potential Function
#----------------------------------------

def harmonic_potential(x):
    """
    Compute the potential energy for a harmonic oscillator.

    :param x: Position
    :return: Potential energy at position x
    """
    return x ** 2 / 2


#----------------------------------------
# Verlet Integration Method
#----------------------------------------

def verlet_integration(energy, is_odd=True):
    """
    Integrate the Schrödinger equation using the Verlet method.

    :param energy: Energy level to integrate over
    :param is_odd: Boolean indicating if the wave function should be odd
    :return: Positions and corresponding wave function values
    """
    position = 0
    wave_function = 0.0 if is_odd else 1.0
    derivative = 1.0 if is_odd else 0.0

    positions = [position]
    wave_functions = [wave_function]

    for _ in range(num_points):
        second_derivative = 2 * (harmonic_potential(position) - energy) * wave_function
        second_derivative_next = 2 * (harmonic_potential(position + step_length) - energy) * wave_function
        wave_function += derivative * step_length + 0.5 * second_derivative * step_length_squared
        derivative += 0.5 * (second_derivative + second_derivative_next) * step_length
        position += step_length
        positions.append(position)
        wave_functions.append(wave_function)

    return positions, wave_functions


#----------------------------------------
# Binary Search for Eigenvalues
#----------------------------------------

def binary_search_energy(e_min, e_max, tolerance=1e-6, is_odd=True):
    """
    Use binary search to find eigenvalues of the system.

    :param e_min: Minimum energy boundary for search
    :param e_max: Maximum energy boundary for search
    :param tolerance: Energy tolerance for convergence
    :param is_odd: Boolean indicating if the wave function should be odd
    :return: Estimated energy eigenvalue
    """
    last_wave_function_value = verlet_integration(e_min, is_odd)[1][-1]
    while abs(e_max - e_min) > tolerance:
        e_mid = (e_max + e_min) / 2
        current_wave_function_value = verlet_integration(e_mid, is_odd)[1][-1]
        if current_wave_function_value * last_wave_function_value > 0:
            e_min = e_mid
            last_wave_function_value = current_wave_function_value
        else:
            e_max = e_mid
    return (e_max + e_min) / 2


#----------------------------------------
# Main plotting execution
#----------------------------------------

analytical_energies = [n + 0.5 for n in range(4)]
calculated_energies = []

sns.set(style="darkgrid")
plt.figure(figsize=(20, 8))

for i in range(4):
    is_odd = i % 2 != 0
    energy = binary_search_energy(analytical_energies[i] - 0.1, analytical_energies[i] + 0.1, is_odd=is_odd)
    calculated_energies.append(energy)
    positions, wave_functions = verlet_integration(energy, is_odd)
    plt.plot(positions, wave_functions, label=f"State {i + 1}, Eigenvalue: {energy:.2f}")


#----------------------------------------
# Plotting and Display Settings
#----------------------------------------

plt.title("Eigenstates of the Harmonic Oscillator")
plt.xlabel("Position")
plt.ylabel("Wave Function")
plt.legend()
plt.show()


#----------------------------------------
# Comparison plot of energies
#----------------------------------------
plt.figure(figsize=(20, 8))
plt.plot(range(1, 5), analytical_energies, "o-", label="Analytical Eigenvalues")
plt.plot(range(1, 5), calculated_energies, "x-", label="Calculated Eigenvalues")
plt.title("Comparison of Analytical and Calculated Eigenvalues")
plt.xlabel("Quantum State")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)
plt.show()