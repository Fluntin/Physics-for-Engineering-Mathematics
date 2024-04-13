import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Given time and amplitude values
times = np.array([0.0, 4.165, 7.8, 11.43, 15.055, 18.685, 22.31, 25.94, 29.565, 33.195])
amplitudes = np.array([1.57079633e+00, 2.06599898e-01, 3.35573740e-02, 5.47040748e-03,
                       8.91863189e-04, 1.45402784e-04, 2.37056674e-05, 3.86479894e-06,
                       6.30095167e-07, 1.02726164e-07])

# Fit a line to the data points
slope, intercept, _, _, _ = linregress(times, np.log(amplitudes))
line = np.exp(intercept) * np.exp(slope * times)

# Create log-y plot
plt.figure(figsize=(8, 6))
plt.scatter(times, amplitudes, color='blue', label='Turning Points')
plt.plot(times, line, color='red', linestyle='--', label='Fitted Line')
plt.yscale('log')
plt.title('Plot of Amplitudes at Turning Points vs Time with Log Scale for y-axis')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

