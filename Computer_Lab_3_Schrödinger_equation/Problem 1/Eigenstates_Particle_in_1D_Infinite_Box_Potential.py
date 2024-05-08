import numpy as np
import matplotlib.pyplot as plt

# Potential energy function for an infinite box, V(x) = 0 inside the box
def V(x):
    return 0.0

# Parameters
Ns = [10, 100, 1000, 10000]
E = 0.5 * np.pi**2  # Trial energy

# Storing results
deltas = []
ns = []

# Run simulation for each N
for N in Ns:
    dx = 1 / N
    dx2 = dx**2

    # Initial conditions
    x = 0
    psi = 0
    dpsi = 1
    psi_end = 0

    # Numerical integration using Euler's method
    for i in range(N):
        d2psi = 2 * (V(x) - E) * psi
        d2psinew = 2 * (V(x + dx) - E) * psi
        psi += dpsi * dx + 0.5 * d2psi * dx2
        dpsi += 0.5 * (d2psi + d2psinew) * dx
        x += dx
        if i == N-1:  # Storing the final value
            psi_end = abs(psi)

    # Store delta and N for plotting
    deltas.append(psi_end)
    ns.append(N)

# Plot configuration
plt.figure(figsize=(10, 6))
plt.loglog(ns, deltas, 'o-', color='blue', linewidth=2, markersize=8, markerfacecolor='orange', label='Simulation Data')
plt.xlabel('Number of points N', fontsize=14, fontweight='bold')
plt.ylabel('Error $|\psi(1)|$', fontsize=14, fontweight='bold')
plt.title('Error Scaling with Mesh Size in Numerical Integration', fontsize=16, fontweight='bold')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
plt.legend()

# Fit a line to the log-log data to illustrate scaling law
coeffs = np.polyfit(np.log(ns), np.log(deltas), 1)
fit_line = np.exp(coeffs[1]) * np.power(ns, coeffs[0])
plt.loglog(ns, fit_line, 'r--', linewidth=2, label=f'Fit Line: slope={coeffs[0]:.2f}')
plt.legend()

# Display the plot with enhanced aesthetics
plt.show()



