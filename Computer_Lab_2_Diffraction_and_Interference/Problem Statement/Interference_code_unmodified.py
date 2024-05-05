# MW 2023-03-20
# Python calculation of wave interference

import numpy as np
import matplotlib.pyplot as plt

wavelength = 5.0
k = 2*np.pi/wavelength
separation = 75.0 # separation of sources
side = 100.0 # sidelength
points = 500 # number of grid points along each side
spacing = side/points 

# positions of wave sources
x1 = side/2+separation/2
y1 = side/2
x2 = side/2-separation/2
y2 = side/2

# array to store amplitude
xi = np.empty([points,points],float)

# calculate amplitudes
for i in range(points):
    y=spacing*i
    for j in range(points):
        x=spacing*j
        r1 = np.sqrt((x-x1)**2+(y-y1)**2)
        r2 = np.sqrt((x-x2)**2+(y-y2)**2)
        xi[i,j] = np.sin(k*r1)+np.sin(k*r2)

# plot
plt.imshow(xi, origin='lower', extent=[-side/2,side/2,-side/2,side/2])
plt.show()

