# Required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------
# Analytical eigenvalues for comparison
#----------------------------------------
analytical_eigenvalues = [n + 0.5 for n in range(1, 5)]

#----------------------------------------
# Simulation parameters
#----------------------------------------
total_points = 20000
simulation_step = 5

#----------------------------------------
# Function to simulate quantum state
#----------------------------------------

def simulate_quantum_state(energy, total_points=10000, is_odd=True, potential_type=0, spatial_step=1):
    """
    Simulate the quantum state using the Verlet integration method.

    :param energy: Energy level for which the state is simulated
    :param total_points: Number of points to simulate
    :param is_odd: Boolean to choose odd (True) or even (False) initial conditions
    :param potential_type: Type of potential (0 for none, 1 for harmonic, 2 for anharmonic)
    :param spatial_step: Step size in spatial grid
    :return: Tuple of position and wavefunction arrays
    """
    delta_x = spatial_step / total_points  # Spatial increment
    delta_x_squared = delta_x**2  # Squared spatial increment

#----------------------------------------
# Function to simulate potential energy
#----------------------------------------

    def potential(x):
        """Potential energy function based on type."""
        if potential_type == 1:
            return x**2 / 2  # Harmonic oscillator
        elif potential_type == 2:
            return x**2 / 2 + x**4  # Anharmonic oscillator
        return 0.0

    #----------------------------------------
    # Initialize variables
    #----------------------------------------
    
    x = 0
    psi = 0 if is_odd else 1
    dpsi = 1 if is_odd else 0
    positions = [x]
    wavefunctions = [psi]

    #----------------------------------------
    # Main simulation loop using Verlet integration
    #----------------------------------------
    
    for _ in range(total_points):
        second_derivative_psi = 2 * (potential(x) - energy) * psi
        second_derivative_psi_next = 2 * (potential(x + delta_x) - energy) * psi
        psi += dpsi * delta_x + 0.5 * second_derivative_psi * delta_x_squared
        dpsi += 0.5 * (second_derivative_psi + second_derivative_psi_next) * delta_x
        x += delta_x
        positions.append(x)
        wavefunctions.append(psi)

    return positions, wavefunctions

#----------------------------------------
# Function to find energy eigenvalue by bisection
#----------------------------------------

def find_energy_by_bisection(energy_min, energy_max, total_points=10000, is_odd=True, potential_type=0, spatial_step=1, tolerance=1e-6):
    """
    Find the energy eigenvalue using bisection method.

    :param energy_min: Lower bound of energy to start search
    :param energy_max: Upper bound of energy to start search
    :return: Estimated energy eigenvalue
    """
    last_psi = simulate_quantum_state(energy_min, total_points, is_odd, potential_type, spatial_step)[1][-1]
    while abs(energy_max - energy_min) > tolerance:
        energy_mid = (energy_max + energy_min) / 2
        psi_mid = simulate_quantum_state(energy_mid, total_points, is_odd, potential_type, spatial_step)[1][-1]
        if psi_mid * last_psi > 0:
            energy_min = energy_mid
            last_psi = psi_mid
        else:
            energy_max = energy_mid
    return (energy_max + energy_min) / 2

energy_ranges = [(0, 2), (4, 6), (10, 12), (19, 21)]
calculated_energies = []
sns.set(style="darkgrid")
plt.figure(figsize=(20, 8))

#----------------------------------------
# Perform energy calculation and plot results
#----------------------------------------

for i, (energy_min, energy_max) in enumerate(energy_ranges, start=1):
    is_odd = i % 2 == 0
    energy = find_energy_by_bisection(energy_min, energy_max, total_points, is_odd, potential_type=2)
    calculated_energies.append(energy)
    positions, wavefunctions = simulate_quantum_state(energy, total_points, is_odd, potential_type=2, spatial_step=1)
    plt.plot(positions, wavefunctions, label=f"State {i}, Eigenvalue: {energy:.2f}")

#----------------------------------------
# Plotting settings
#----------------------------------------

plt.title("Eigenstates of the Anharmonic Oscillator", fontsize=16, fontweight='bold')
plt.xlabel("x", fontsize=14, fontweight='bold')
plt.ylabel("$\psi$ $(x)$", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()