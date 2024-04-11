import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

N = 10 # antal svängningspunkter
dTheta = np.pi/(2*N)
turnAmp = np.ones((N,2))
turnAmp[0,0] = 0
turnAmp[0,1] = np.pi/2
gamma = 8/8 # damping coefficient
timeThreshold = 500

dt = 0.005 # time step
dt2 = dt/2 # half time step
t = 0 # start time

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

def f(theta, p, t):
  accel = -omega02*np.sin(theta) # pendulum
  accel += -gamma*p # damping
  return accel

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

    if (p >= 0 and p_prev < 0) or (p <= 0 and p_prev > 0):
        k = k+1
        turnAmp[k,:] = [t, np.abs(theta)]

        if k == N-1:
            return False
    
    if theta>np.pi: theta -= 2*np.pi
    if theta<-np.pi: theta += 2*np.pi
    if t > timeThreshold and theta>0: return True #När systemet är 'overdamped'

theta = np.pi/2 # initial angular position
p = 0. # initial angular velocity
# model parameters (set m=g=L=1)
omega0 = 8/8 # natural frequency
omega02 = omega0**2
#omega = 2/3 # drive frequency
#A = 1.0 # amplitude of drive force
position = [] # list to store angular position
momentum = []

step(t, p, theta, position, momentum)

#tau beräkning med interpolation
interp = sc.interpolate.interp1d(turnAmp[0:2,0],turnAmp[0:2,1], kind="linear")
print("Times at first two turning points:", turnAmp[0:2, 0])
print("Amplitudes at first two turning points:", turnAmp[0:2, 1])

plt.figure("Figure 1", figsize=(8, 6))  # Set the figure size for better presentation
plt.yscale('log')
plt.scatter(turnAmp[:, 0], np.log(turnAmp[:, 1]), color='blue', label='Turning Points')
plt.plot(np.linspace(turnAmp[0, 0], turnAmp[1, 0], 100), interp(np.linspace(turnAmp[0, 0], turnAmp[1, 0], 100)), color='red', label='Interpolation')
plt.title("Logarithm of the Amplitude at the Turning Points vs Time", fontsize=14)  # Increase title font size
plt.xlabel("Time", fontsize=12)  # Increase x-axis label font size
plt.ylabel("Logarithm of the Amplitude at the Turning Points", fontsize=12)  # Increase y-axis label font size
plt.legend()  # Show legend
plt.grid(True, which="both", ls="--", alpha=0.5)  # Add grid lines with dashed style and reduced transparency
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.show()



M = 100 #antal punkter inpterpolerade
timeVec = np.linspace(turnAmp[0,0],turnAmp[1,0],M)
timeStep = (turnAmp[1,0]-turnAmp[0,0])/M

interpValues = interp(timeVec)
i = 0

# Interpolation to find characteristic time where amplitude reduces to half its initial value
tau=-np.log(2)/((np.log(turnAmp[1,1])- np.log(turnAmp[0,1]))/((turnAmp[1,0])- (turnAmp[0,0])))
print(f"Characteristic decay time (tau) where amplitude reduces to half its initial value: {tau} seconds")


prevTau = 0
while i < M-2:
  i += 1
  tau = timeStep*i
  if interpValues[i] >= turnAmp[0,1]-np.log(2) and interpValues[i+1] <= turnAmp[0,1]-np.log(2):
    print(f"Calculated time from interpolation: {tau} seconds")
    break


#olika gamma värden
i = 0
while True:
  gamma = i*0.1
  stuck = step(t, p, theta, position, momentum)

  if stuck == True:
    print(f"The system becomes overdamped at a damping coefficient (Gamma) of: {gamma}")
    break
  i += 1

#räkna och plotta graferna för gamma och amplituden vid första svängningen
antalSvängningar = int(gamma/0.1 - 1)
gammaAmp = np.zeros((antalSvängningar, 2))
gammaInvAmp = np.zeros((antalSvängningar, 2))

for i in range(antalSvängningar):
  gamma = 0.1 * (i+1)
  p = 0
  t = 0
  theta = np.pi/2
  while True:
    p_prev = p
    
    # 1.38 for tau
    
    theta, p, t = rk4(theta, p, t)
    # energy
    #H = 0.5*p**2 + 1 - np.cos(theta)
    # position
    position.append(theta)
    momentum.append(p)

    if (p >= 0 and p_prev < 0) or (p <= 0 and p_prev > 0):
      gammaAmp[i, :] = [gamma, abs(theta)]
      gammaInvAmp[i, :] = [1/gamma, abs(theta)]
      break
    
    if theta>np.pi: theta -= 2*np.pi
    if theta<-np.pi: theta += 2*np.pi
    if gamma == 2: 
      gammaAmp[i, :] = [gamma, 0]
      gammaInvAmp[i, :] = [1/gamma, 0]
      break

# Printing the results in a formatted way
print(f"{'Damping Coefficient (Gamma)':^30} | {'Amplitude at First Turn':^30}")
print("=" * 63)

for row in gammaAmp:
    gamma, amplitude = row
    print(f"{gamma:^30.2f} | {amplitude:^30.2e}")

plt.figure("Absolute Amplitude vs Damping Coefficient (\u03B3)", figsize=(8, 6))  # Set the figure size for better presentation
plt.scatter(gammaAmp[:, 0], gammaAmp[:, 1], color='green')  # Set the color of the scatter plot
plt.title("Absolute Value of the Amplitude at the First Turn vs Damping Coefficient (\u03B3)", fontsize=14)  # Increase title font size
plt.xlabel("Damping Coefficient (\u03B3)", fontsize=12)  # Increase x-axis label font size
plt.ylabel("Absolute Value of the Amplitude at the First Turn", fontsize=12)  # Increase y-axis label font size
plt.grid(True, which="both", ls="--", alpha=0.5)  # Add grid lines with dashed style and reduced transparency
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.show()

plt.figure("Amplitude vs Inverse Damping Coefficient (1/\u03B3)", figsize=(8, 6))  # Set the figure size for better presentation
plt.scatter(gammaInvAmp[:, 0], gammaInvAmp[:, 1], color='orange')  # Set the color of the scatter plot
plt.title("Amplitude at the Turning Points vs Inverse Damping Coefficient (1/\u03B3)", fontsize=14)  # Increase title font size
plt.xlabel("Inverse Damping Coefficient (1/\u03B3)", fontsize=12)  # Increase x-axis label font size
plt.ylabel("Amplitude at the Turning Points", fontsize=12)  # Increase y-axis label font size
plt.grid(True, which="both", ls="--", alpha=0.5)  # Add grid lines with dashed style and reduced transparency
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.show()

