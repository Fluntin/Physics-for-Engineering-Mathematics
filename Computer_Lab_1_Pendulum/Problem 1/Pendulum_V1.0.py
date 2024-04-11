import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# Constants
num_points = 100  # Number of points to simulate
angular_step = np.pi / (2 * num_points)  # Step size for the initial angular position
time_step = 0.005  # Time step for simulation
half_time_step = time_step / 2  # Half of the time step

# Arrays to store results
approximations = np.zeros((num_points, 6))  # Store approximation results
analytical_solutions = np.zeros(num_points)  # Store analytical solutions for comparison

# Simulation parameters
start_time = 0  # Starting time of the simulation

def runge_kutta_4th_order(theta, angular_velocity, current_time):
    """
    Runge-Kutta 4th order method for approximating solutions of ODEs.
    """
    k1_theta = time_step * angular_velocity
    k1_velocity = time_step * calculate_acceleration(theta, angular_velocity, current_time)
    
    k2_theta = time_step * (angular_velocity + k1_velocity / 2)
    k2_velocity = time_step * calculate_acceleration(theta + k1_theta / 2, angular_velocity + k1_velocity / 2, current_time + half_time_step)
    
    k3_theta = time_step * (angular_velocity + k2_velocity / 2)
    k3_velocity = time_step * calculate_acceleration(theta + k2_theta / 2, angular_velocity + k2_velocity / 2, current_time + half_time_step)
    
    k4_theta = time_step * (angular_velocity + k3_velocity)
    k4_velocity = time_step * calculate_acceleration(theta + k3_theta, angular_velocity + k3_velocity, current_time + time_step)
    
    theta += (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta) / 6
    angular_velocity += (k1_velocity + 2*k2_velocity + 2*k3_velocity + k4_velocity) / 6
    current_time += time_step
    
    return theta, angular_velocity, current_time

def calculate_acceleration(theta, angular_velocity, current_time):
    """
    Calculates the acceleration of the pendulum based on its current state.
    """
    natural_frequency_squared = 1  # omega^2, with m=g=L=1
    acceleration = -natural_frequency_squared * np.sin(theta)  # Acceleration due to gravity
    return acceleration

def simulate_pendulum_motion(current_time, angular_velocity, initial_theta, position_list, momentum_list):
    """
    Simulates the pendulum's motion until it reaches the top, recording position and momentum.
    """
    while True:
        initial_theta, angular_velocity, current_time = runge_kutta_4th_order(initial_theta, angular_velocity, current_time)
        
        position_list.append(initial_theta)
        momentum_list.append(angular_velocity)

        if angular_velocity >= 0:  # Pendulum reaches the top position
            period = current_time * 2
            return period
        
        # Adjust theta to be within [-pi, pi] range
        if initial_theta > np.pi: initial_theta -= 2 * np.pi
        if initial_theta < -np.pi: initial_theta += 2 * np.pi

for i in range(num_points):
    initial_theta = i * angular_step  # Initial angular position
    initial_angular_velocity = 0.  # Initial angular momentum
    
    # Lists to store the pendulum's position and momentum for plotting
    position_list = []
    momentum_list = []

    period = simulate_pendulum_motion(start_time, initial_angular_velocity, initial_theta, position_list, momentum_list)
    
    # Store results in approximations array
    approximations[i, :5] = [initial_theta, 2*np.pi, period, 2*np.pi*(1+initial_theta**2/16), 2*np.pi*(1+initial_theta**2/16+11*initial_theta**4/3072)]
    
    # Calculate the analytical solution
    analytical_solutions[i] = np.sqrt(2) * sc.integrate.quad(lambda x: 1/np.sqrt(np.cos(x) - np.cos(initial_theta)), -initial_theta, initial_theta)[0]

# Adding the analytical solutions to the approximations for comparison
approximations[:,5] = analytical_solutions

# Further adjusting the plot aesthetics and ensuring the annotation is clearly visible
plt.figure(figsize=(15,5))
plt.style.use('seaborn-whitegrid')

plt.plot(approximations[1:,0], approximations[1:,1], linestyle='-', color='#0055AA', linewidth=1.5, label='Harmonic Oscillator')
plt.plot(approximations[1:,0], approximations[1:,3], linestyle='--', color='#5599FF', linewidth=1.5, label='Second Order Power Series')
plt.plot(approximations[1:,0], approximations[1:,4], linestyle='-.', color='#55CC55', linewidth=1.5, label='Fourth Order Power Series')
plt.plot(approximations[1:,0], approximations[1:,5], linestyle=':', color='#FFAA55', linewidth=1.5, label='Analytical Solution')

plt.scatter(approximations[1:,0], approximations[1:,2], s=10, color="#DD3333", label='Simulation Result', zorder=5)

plt.title("Time vs Initial Angular Position", fontsize=15)
plt.xlabel("Initial Angle (Radians)", fontsize=13)
plt.ylabel("Period (Seconds)", fontsize=13)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='#aaaaaa')

# Adjusting the annotation position for better visibility
plt.annotate(f'Calculated with Δt = {time_step} s', xy=(0.7, 0.1), xycoords='axes fraction', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor='#bbbbbb', facecolor='white', alpha=0.9))

plt.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.95)

plt.tight_layout()
plt.show()