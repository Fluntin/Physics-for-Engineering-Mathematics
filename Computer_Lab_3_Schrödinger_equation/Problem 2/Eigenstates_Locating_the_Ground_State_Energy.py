import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10000            # Number of mesh points
dx = 1.0 / N         # Step length
dx2 = dx ** 2        # Step length squared

# Function defining the potential inside the box
def V(x):
    return 0.0  # Potential is zero inside the box

# Function to integrate the Schrodinger equation using Euler's method
def integrate(E):
    psi = 0.0  # Wave function at initial position
    dpsi = 1.0  # Derivative of wave function at initial position
    x_tab = [0]  # List to store positions for plot
    psi_tab = [psi]  # List to store wave function for plot

    for i in range(N):
        x = i * dx
        d2psi = 2 * (V(x) - E) * psi
        d2psinew = 2 * (V(x + dx) - E) * psi
        psi += dpsi * dx + 0.5 * d2psi * dx2
        dpsi += 0.5 * (d2psi + d2psinew) * dx
        x_tab.append(x + dx)
        psi_tab.append(psi)

    return x_tab, psi_tab, psi

# Binary search to find the eigenvalues
def find_eigenvalue(E_min, E_max, tolerance):
    while E_max - E_min > tolerance:
        E_mid = (E_min + E_max) / 2
        x, psi, psi_end = integrate(E_mid)
        if psi_end * integrate(E_min)[2] < 0:  # Checking change in sign at the boundary
            E_max = E_mid
        else:
            E_min = E_mid
    return E_mid, psi

# Initialize energy search boundaries and tolerance
E_min, E_max = 2, 50  # Adjust based on expected energy levels
tolerance = 1e-10

# Find and plot the first four eigenvalues and their eigenfunctions
eigenvalues = []
eigenstates = []
for i in range(4):
    E, psi = find_eigenvalue(E_min, E_max, tolerance)
    eigenvalues.append(E)
    eigenstates.append(psi)
    E_min = E + 10  # Reset lower boundary for the next eigenvalue search
    E_max = E + 60  # Reset upper boundary for a broader range

plt.figure(figsize=(10, 8))
for i, psi in enumerate(eigenstates):
    plt.plot(np.linspace(0, 1, N + 1), psi, label=f'Eigenstate {i + 1} with E = {eigenvalues[i]:.6f}')
plt.title("First four eigenstates for a particle in a box")
plt.xlabel("x")
plt.ylabel("$\psi(x)$")
plt.legend()
plt.grid(True)
plt.show()