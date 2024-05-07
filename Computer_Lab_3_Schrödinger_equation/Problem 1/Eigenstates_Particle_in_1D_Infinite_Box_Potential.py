# -*- coding: utf-8 -*-
# Python simulation of a particle in a 1D infinite box potential
# Integrate time-independent Schrödinger equation using the Verlet method
# Boundary conditions are found by shooting
# MW 220513

import numpy as np
import matplotlib.pyplot as plt

# Define the potential energy function for a particle in an infinite box
def V(x):
    return 0.0  # Potential inside the box is zero

# Constants
E = 0.5 * np.pi**2  # Energy, set to the ground state energy for n=1

# Parameters
N = 10000  # Number of mesh points, increased for better accuracy
dx = 1 / N  # Step length
dx2 = dx**2  # Step length squared

# Initial conditions
x = 0  # Initial position
psi = 0  # Wave function at initial position, psi(0) = 0 (boundary condition)
dpsi = 1  # Derivative of wave function at initial position

# Lists to store position and wave function values for plotting
x_tab = [x]
psi_tab = [psi]

# Integrate using the modified Verlet method
for i in range(N):
    d2psi = 2 * (V(x) - E) * psi
    psi_new = psi + dpsi * dx + 0.5 * d2psi * dx2
    d2psi_new = 2 * (V(x + dx) - E) * psi_new
    dpsi += 0.5 * (d2psi + d2psi_new) * dx
    psi = psi_new
    x += dx
    x_tab.append(x)
    psi_tab.append(psi)

# Output the final value of psi to check the boundary condition at x = 1
print(f'E = {E}, psi(x = 1) = {psi}')

# Plotting the wave function
plt.figure(figsize=(8, 6))
plt.plot(x_tab, psi_tab, label='Wave function $\psi(x)$', color='red')
plt.xlabel('x')
plt.ylabel('$\psi(x)$')
plt.title('Wave Function $\psi(x)$ for a Particle in a 1D Infinite Box')
plt.grid(True)
plt.legend()
plt.show()



