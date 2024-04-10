
# MW 2022-03-23
# Python simulation of damped driven pendulum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# time parameters 
dt = 0.01             # time step
dt2 = dt/2            # half time step
t = 0  	              # start time 

# initial conditions
theta = np.pi/2   # initial angular position 
p = 0.                 # initial angular velocity

# model parameters (set m=g=L=1)
omega0 = 1           # natural frequency
omega02 = omega0**2       
#gamma = 3/8          # damping coefficient
#omega = 2/3          # drive frequency
#A = 1.0              # amplitude of drive force

position = []         # list to store angular position
momentum = []         # list to store angular momentum

#-----------------------------------

theta_previous = theta  # Store the previous theta to compare changes
theta_increasing = False  # Flag to track when theta starts increasing
half_period_detected = False  # Flag to ensure the half-period is only recorded once
half_period_time = 0  # Time at which the half-period occurs
full_periods = []  # List to store full period times for different initial conditions

#-----------------------------------

# set up the figure and the plot element to animate
fig = plt.figure(figsize=(10,14),dpi=80)
ax1 = plt.subplot(211, aspect='equal', autoscale_on=False, xlim=(-1.1,1.1), ylim=(-1.1,1.1))
pendulum, = ax1.plot([], [], c='r', lw=10)
ax1.axis('off')
ax2 = plt.subplot(212, aspect='equal', autoscale_on=False, xlim=(-np.pi, np.pi), ylim=(-3, 3))
phaseportrait, = ax2.plot([], [], 'bo', markersize=0.5)# c='black', lw=0)
ax2.set_xlabel(r'$\theta$')
ax2.set_ylabel(r'p')

def f(theta, p, t):
    accel = -omega02*np.sin(theta) # pendulum
    #accel += -gamma*p              # damping  
    #accel += A*np.cos(omega*t)     # drive force
    return accel

def rk4(x, v, t):
    xk1 = dt*v
    vk1 = dt*f(x, v, t)
    xk2 = dt*(v+vk1/2)
    vk2 = dt*f(x+xk1/2, v+vk1/2, t+dt/2)
    xk3 = dt*(v+vk2/2)
    vk3 = dt*f(x+xk2/2, v+vk2/2, t+dt/2)
    xk4 = dt*(v+vk3)
    vk4 = dt*f(x+xk3, v+vk3, t+dt)
    x += (xk1+2*xk2+2*xk3+xk4)/6
    v += (vk1+2*vk2+2*vk3+vk4)/6
    t += dt
    return x, v, t

def step():
    global xx, yy, t, p, theta, H, time, ene, position, momentum, theta_previous, theta_increasing, half_period_detected, half_period_time, full_periods

    theta, p, t = rk4(theta, p, t)

    # Detect change from decreasing to increasing theta
    if not theta_increasing and theta > theta_previous:
        if not half_period_detected:  # Ensure we only record the first occurrence
            half_period_time = t
            half_period_detected = True
            full_period = 2 * half_period_time  # Calculate full period
            full_periods.append(full_period)  # Store the calculated full period

    # Update the position of the pendulum
    xx = (0, np.sin(theta))
    yy = (0, -np.cos(theta))

    position.append(theta)
    momentum.append(p)

    # Ensuring theta is within the range [-pi, pi]
    if theta > np.pi: theta -= 2 * np.pi
    if theta < -np.pi: theta += 2 * np.pi

    # Update previous_theta for the next step
    theta_previous = theta

# Initialization function
def init():
    pendulum.set_data([], [])
    phaseportrait.set_data([], [])
    return pendulum, phaseportrait, 


# Animation function
def animate(i):
    step()
    pendulum.set_data(xx, yy)
    phaseportrait.set_data(position, momentum)
    print(theta)
    print (full_periods)
    return pendulum, phaseportrait, 

anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=1000, interval=1, blit=True, repeat=True)

plt.show()

