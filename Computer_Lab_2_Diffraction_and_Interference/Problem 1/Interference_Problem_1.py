import numpy as np
import matplotlib.pyplot as plt

wavelength = 5.0
k = 2*np.pi/wavelength
separation = 40.0     # separation of sources
side = 100.0          # sidelength
points = 500          # number of grid points along each side
spacing = side/points 

# Positions of wave sources:
# Source 1
x1 = side/2+separation/2
y1 = side/2

# Source 2
x2 = side/2-separation/2
y2 = side/2

# Source 3
x3 = side/2
y3 = side/2

# Array to store amplitude
xi = np.empty([points,points],float)

# Calculate amplitudes
for i in range(points):
    y=spacing*i
    for j in range(points):
        x=spacing*j
        r1 = np.sqrt((x-x1)**2+(y-y1)**2)
        r2 = np.sqrt((x-x2)**2+(y-y2)**2)
        r3 = np.sqrt((x-x3)**2+(y-y3)**2)
        xi[i,j] = np.sin(k*r1)+np.sin(k*r2)+np.sin(k*r3)

# Plot
plt.imshow(xi, origin='lower', extent=[-side/2,side/2,-side/2,side/2])
plt.show()

