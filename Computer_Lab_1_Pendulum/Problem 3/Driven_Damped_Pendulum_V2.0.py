import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Time parameters
time_step = 0.005  # Time step
half_time_step = time_step / 2  # Half time step
current_time = 0  # Start time

# Initial conditions
initial_angular_position = 0.  # Initial angular position
initial_angular_velocity = 0.  # Initial angular velocity

# Model parameters (set m=g=L=1)
natural_frequency = 1  # Natural frequency
natural_frequency_squared = natural_frequency ** 2
damping_coefficient = 3 / 8  # Damping coefficient
drive_frequency = 2 / 3  # Drive frequency
drive_force_amplitudes = [0.1, 0.2, 0.4, 0.8]  # Amplitude of drive force

# Lists to store data
angular_position_data = []  # List to store angular position
angular_momentum_data = []  # List to store angular momentum
amplitude_data = []  # List of amplitude
amplitude_at_steady_state = []  # List of amplitude at steady state

def calculate_acceleration(theta, p, t):
    # Calculate acceleration
    acceleration = -natural_frequency_squared * np.sin(theta)  # Pendulum
    acceleration += -damping_coefficient * p  # Damping
    acceleration += drive_amplitude * np.cos(drive_frequency * t)  # Drive force
    return acceleration

def runge_kutta_fourth_order(x, v, t):
    # Runge-Kutta 4th order method
    xk1 = time_step * v
    vk1 = time_step * calculate_acceleration(x, v, t)
    xk2 = time_step * (v + vk1 / 2)
    vk2 = time_step * calculate_acceleration(x + xk1 / 2, v + vk1 / 2, t + time_step / 2)
    xk3 = time_step * (v + vk2 / 2)
    vk3 = time_step * calculate_acceleration(x + xk2 / 2, v + vk2 / 2, t + time_step / 2)
    xk4 = time_step * (v + vk3)
    vk4 = time_step * calculate_acceleration(x + xk3, v + vk3, t + time_step)
    x += (xk1 + 2 * xk2 + 2 * xk3 + xk4) / 6
    v += (vk1 + 2 * vk2 + 2 * vk3 + vk4) / 6
    t += time_step
    return x, v, t

def simulation_step(t, p, theta):
    # Perform simulation step
    while True:
        p_prev = p
        theta, p, t = runge_kutta_fourth_order(theta, p, t)

        # Store data
        angular_position_data.append(theta)
        angular_momentum_data.append(p)

        # Check for steady state
        if t > 500:
            if (p >= 0 and p_prev < 0) or (p <= 0 and p_prev > 0):
                return np.abs(theta)

        # Normalize theta
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

# Perform simulation for different amplitudes
for amplitude in drive_force_amplitudes:
    drive_amplitude = amplitude
    amplitude_at_steady_state.append(simulation_step(current_time, initial_angular_velocity, initial_angular_position))

# Plot results
plt.figure(figsize=(8, 6))  # Adjust figure size
plt.scatter(drive_force_amplitudes, amplitude_at_steady_state, color='blue', marker='o', label='Data')  # Customize scatter plot
plt.title("Amplitude at Steady State vs. Amplitude (A)", fontsize=16)  # Add title with increased font size
plt.xlabel("Amplitude (A)", fontsize=14)  # Add x-axis label with increased font size
plt.ylabel("Amplitude at Steady State", fontsize=14)  # Add y-axis label with increased font size
plt.xticks(fontsize=12)  # Increase font size of x-axis ticks
plt.yticks(fontsize=12)  # Increase font size of y-axis ticks
plt.grid(True, linestyle='--', alpha=0.5)  # Add grid lines with dashed style and transparency
plt.legend(fontsize=12)  # Add legend with increased font size
plt.tight_layout()  # Adjust layout to prevent overlapping elements
plt.show()

