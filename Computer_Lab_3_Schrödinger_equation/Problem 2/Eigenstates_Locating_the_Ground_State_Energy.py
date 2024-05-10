import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------
# Parameters
#----------------------------------------

N = 10000            # Number of mesh points
dx = 1.0 / N         # Step length
dx2 = dx ** 2        # Step length squared

#----------------------------------------
# Function to integrate the Schrodinger equation using modified Verlet method
#----------------------------------------

def verlet_integration(E):
    psi = 0.0  # Wave function at initial position
    dpsi = 1.0  # Derivative of wave function at initial position
    x_tab = [0]  # List to store positions for plot
    psi_tab = [psi]  # List to store wave function for plot
    x = 0

    for i in range(N):
        d2psi = -2 * E * psi  # Since V(x) = 0, only energy term remains
        d2psinew = -2 * E * psi  # No dependence on x for potential
        psi += dpsi * dx + 0.5 * d2psi * dx2
        dpsi += 0.5 * (d2psi + d2psinew) * dx
        x += dx
        x_tab.append(x)
        psi_tab.append(psi)

    return x_tab, psi_tab

#----------------------------------------
# Binary search to find the eigenvalues
#----------------------------------------

def binary_search(E_min, E_max, tolerance=1e-6):
    psi_prev = verlet_integration(E_min)[1][-1]
    while abs(E_max - E_min) > tolerance:
        E_mid = (E_max + E_min) / 2
        psi = verlet_integration(E_mid)[1][-1]
        if psi * psi_prev > 0:
            E_min = E_mid
            psi_prev = psi
        else:
            E_max = E_mid
    return (E_max + E_min) / 2

#----------------------------------------
# Main Execution and Plotting
#----------------------------------------


E_analytical = [(n**2 * np.pi**2) / 2 for n in range(1, 5)]
E_calculated = []

sns.set(style="darkgrid")
plt.figure(figsize=(20, 8))
for i in range(1, 5):
    E = binary_search(E_analytical[i - 1] - 0.1, E_analytical[i - 1] + 0.1)
    E_calculated.append(E)
    x, psi = verlet_integration(E)
    plt.plot(x, psi, label=f"State {i}, Eigenvalue: {E:.2f}")
    

plt.title("Eigenstates for a Particle in a Box", fontsize=16, fontweight='bold')
plt.xlabel("x", fontsize=14, fontweight='bold')
plt.ylabel("$\psi(x)$", fontsize=14, fontweight='bold')
plt.legend()
plt.show()

#----------------------------------------
# Energy Comparison Plot
#----------------------------------------

plt.figure(figsize=(20, 8))
plt.plot(range(1, 5), E_analytical, "o-", label="Analytical Eigenvalues")
plt.plot(range(1, 5), E_calculated, "x-", label="Calculated Eigenvalues")
plt.title("Comparison of Analytical and Calculated Eigenvalues", fontsize=16, fontweight='bold')
plt.xlabel("Quantum State", fontsize=14, fontweight='bold')
plt.ylabel("Energy", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()