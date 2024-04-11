import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# Number of oscillation points
num_oscillation_points = 10
d_theta = np.pi / (2 * num_oscillation_points)
turn_amp = np.ones((num_oscillation_points, 2))
turn_amp[0, 0] = 0
turn_amp[0, 1] = np.pi / 2
damping_coefficient = 8 / 8
time_threshold = 500

dt = 0.005  # time step
dt_half = dt / 2  # half time step
t = 0  # start time

def rk4(x, v, t):
    xk1 = dt * v
    vk1 = dt * f(x, v, t)
    xk2 = dt * (v + vk1 / 2)
    vk2 = dt * f(x + xk1 / 2, v + vk1 / 2, t + dt_half)
    xk3 = dt * (v + vk2 / 2)
    vk3 = dt * f(x + xk2 / 2, v + vk2 / 2, t + dt_half)
    xk4 = dt * (v + vk3)
    vk4 = dt * f(x + xk3, v + vk3, t + dt)
    x += (xk1 + 2 * xk2 + 2 * xk3 + xk4) / 6
    v += (vk1 + 2 * vk2 + 2 * vk3 + vk4) / 6
    t += dt
    return x, v, t

def f(theta, p, t):
    accel = -omega_0_square * np.sin(theta)  # pendulum
    accel += -damping_coefficient * p  # damping
    return accel

def step(t, p, theta, position, momentum):
    k = 0
    while True:
        p_prev = p

        theta, p, t = rk4(theta, p, t)
        # energy
        # H = 0.5 * p**2 + 1 - np.cos(theta)
        # position
        position.append(theta)
        momentum.append(p)

        if (p >= 0 and p_prev < 0) or (p <= 0 and p_prev > 0):
            k = k + 1
            turn_amp[k, :] = [t, np.abs(theta)]

            if k == num_oscillation_points - 1:
                return False

        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi
        if t > time_threshold and theta > 0:
            return True  # When the system is 'overdamped'

# Initial angular position and velocity
initial_angular_position = np.pi / 2
initial_angular_velocity = 0.0

# Model parameters (set m=g=L=1)
omega_0 = 8 / 8  # natural frequency
omega_0_square = omega_0 ** 2

# Lists to store angular position and momentum
position = []
momentum = []

step(t, initial_angular_velocity, initial_angular_position, position, momentum)

# Interpolation for tau calculation
interp = sc.interpolate.interp1d(turn_amp[0:2, 0], turn_amp[0:2, 1], kind="linear")
print("Times at first two turning points:", turn_amp[0:2, 0])
print("Amplitudes at first two turning points:", turn_amp[0:2, 1])

# Plotting
plt.figure("Figure 1", figsize=(8, 6))
plt.yscale('log')
plt.scatter(turn_amp[:, 0], np.log(turn_amp[:, 1]), color='blue', label='Turning Points')
plt.plot(np.linspace(turn_amp[0, 0], turn_amp[1, 0], 100), interp(np.linspace(turn_amp[0, 0], turn_amp[1, 0], 100)), color='red', label='Interpolation')
plt.title("Logarithm of the Amplitude at the Turning Points vs Time", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Logarithm of the Amplitude at the Turning Points", fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

# Calculation of characteristic time where amplitude reduces to half its initial value
M = 100  # Number of interpolated points
time_vec = np.linspace(turn_amp[0, 0], turn_amp[1, 0], M)
time_step = (turn_amp[1, 0] - turn_amp[0, 0]) / M
interp_values = interp(time_vec)

tau = -np.log(2) / ((np.log(turn_amp[1, 1]) - np.log(turn_amp[0, 1])) / ((turn_amp[1, 0]) - (turn_amp[0, 0])))
print(f"Characteristic decay time (tau) where amplitude reduces to half its initial value: {tau} seconds")

prev_tau = 0
i = 0
while i < M - 2:
    i += 1
    tau = time_step * i
    if interp_values[i] >= turn_amp[0, 1] - np.log(2) and interp_values[i + 1] <= turn_amp[0, 1] - np.log(2):
        print(f"Calculated time from interpolation: {tau} seconds")
        break

# Different gamma values
i = 0
while True:
    damping_coefficient = i * 0.1
    stuck = step(t, initial_angular_velocity, initial_angular_position, position, momentum)

    if stuck:
        print(f"The system becomes overdamped at a damping coefficient (Gamma) of: {damping_coefficient}")
        break
    i += 1

# Calculation and plotting of graphs for gamma and amplitude at the first turn
num_gamma_values = int(damping_coefficient / 0.1 - 1)
gamma_amp = np.zeros((num_gamma_values, 2))
gamma_inv_amp = np.zeros((num_gamma_values, 2))

for i in range(num_gamma_values):
    damping_coefficient = 0.1 * (i + 1)
    initial_angular_velocity = 0
    t = 0
    initial_angular_position = np.pi / 2
    while True:
        p_prev = initial_angular_velocity

        initial_angular_position, initial_angular_velocity, t = rk4(initial_angular_position, initial_angular_velocity, t)

        position.append(initial_angular_position)
        momentum.append(initial_angular_velocity)

        if (initial_angular_velocity >= 0 and p_prev < 0) or (initial_angular_velocity <= 0 and p_prev > 0):
            gamma_amp[i, :] = [damping_coefficient, abs(initial_angular_position)]
            gamma_inv_amp[i, :] = [1 / damping_coefficient, abs(initial_angular_position)]
            break

        if initial_angular_position > np.pi:
            initial_angular_position -= 2 * np.pi
        if initial_angular_position < -np.pi:
            initial_angular_position += 2 * np.pi
        if damping_coefficient == 2:
            gamma_amp[i, :] = [damping_coefficient, 0]
            gamma_inv_amp[i, :] = [1 / damping_coefficient, 0]
            break

# Printing the results
print(f"{'Damping Coefficient (Gamma)':^30} | {'Amplitude at First Turn':^30}")
print("=" * 63)
for row in gamma_amp:
    damping, amplitude = row
    print(f"{damping:^30.2f} | {amplitude:^30.2e}")

# Plotting for gamma and amplitude at the first turn
plt.figure("Absolute Amplitude vs Damping Coefficient (\u03B3)", figsize=(8, 6))
plt.scatter(gamma_amp[:, 0], gamma_amp[:, 1], color='green')
plt.title("Absolute Value of the Amplitude at the First Turn vs Damping Coefficient (\u03B3)", fontsize=14)
plt.xlabel("Damping Coefficient (\u03B3)", fontsize=12)
plt.ylabel("Absolute Value of the Amplitude at the First Turn", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure("Amplitude vs Inverse Damping Coefficient (1/\u03B3)", figsize=(8, 6))
plt.scatter(gamma_inv_amp[:, 0], gamma_inv_amp[:, 1], color='orange')
plt.title("Amplitude at the Turning Points vs Inverse Damping Coefficient (1/\u03B3)", fontsize=14)
plt.xlabel("Inverse Damping Coefficient (1/\u03B3)", fontsize=12)
plt.ylabel("Amplitude at the Turning Points", fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()