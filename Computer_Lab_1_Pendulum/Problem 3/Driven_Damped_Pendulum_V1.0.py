import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# time parameters
dt = 0.005 # time step
dt2 = dt/2 # half time step
t = 0 # start time
# initial conditions
theta = 0. # initial angular position
p = 0. # initial angular velocity
# model parameters (set m=g=L=1)
omega0 = 1 # natural frequency
omega02 = omega0**2
gamma = 3/8 # damping coefficient
omega = 2/3 # drive frequency
Aval = [0.1, 0.2, 0.4, 0.8] # amplitude of drive force
position = [] # list to store angular position
momentum = [] # list to store angular momentum
amp = [] # list of amplitude in yy
turnAmp = []

def f(theta, p, t):
  accel = -omega02*np.sin(theta) # pendulum
  accel += -gamma*p # damping
  accel += A*np.cos(omega*t) # drive force
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


def step(t, p, theta, position, momentum):
  k = 0
  while True:
    p_prev = p
    
    theta, p, t = rk4(theta, p, t)
    # energy
    #H = 0.5*p**2 + 1 - np.cos(theta)
    # position
    position.append(theta)
    momentum.append(p)

    if t > 500:
        if (p >= 0 and p_prev < 0) or (p <= 0 and p_prev > 0):
            return np.abs(theta)

    if theta>np.pi: theta -= 2*np.pi
    if theta<-np.pi: theta += 2*np.pi

for i in range(len(Aval)):
   A = Aval[i]
   turnAmp.append(step(t, p, theta, position, momentum))
print(turnAmp)
plt.scatter(Aval,turnAmp)
plt.title("Amplitude at the steady state vs A")
plt.xlabel("A")
plt.ylabel("Amplitude at the steady state")
plt.show()