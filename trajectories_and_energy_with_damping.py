import num_method
import eom
import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81
l1 = 10.0
l2 = 1.0
m1 = 1.0
m2 = 10.0
b = 0.5  # Damping coefficient (adjust this value to change damping strength)


# Initial conditions: [theta1, theta2, omega1, omega2]
y0 = [np.pi / 2, np.pi / 2 + 0.01, 0, 0]

# Time settings
t_max = 20
dt = 0.01
t_eval = np.arange(0, t_max, dt)

# Calculate energy
def compute_energy(states, m1, m2, l1, l2, g):
    theta1, theta2, omega1, omega2 = states.T
    
    # Kinetic energy
    T1 = 0.5 * m1 * (l1 * omega1)**2
    T2 = 0.5 * m2 * ((l1 * omega1)**2 + (l2 * omega2)**2 + 
                     2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2))
    T = T1 + T2
    
    # Potential energy (reference at bottom position)
    V1 = m1 * g * l1 * (1 - np.cos(theta1))
    V2 = m2 * g * (l1 * (1 - np.cos(theta1)) + l2 * (1 - np.cos(theta2)))
    V = V1 + V2
    
    return T + V

# Integrate over time using RK4
states = [y0]
y = np.array(y0)
for t in t_eval:
    y = num_method.rk4_step(eom.double_pendulum_with_damping, t, y, dt, m1, m2, l1, l2, b)
    states.append(y)

states = np.array(states)

# Extract states
theta1 = states[:, 0]
theta2 = states[:, 1]
omega1 = states[:, 2]
omega2 = states[:, 3]
energy = compute_energy(states, m1, m2, l1, l2, g)

# Create figure with subplots
fig = plt.figure(figsize=(12, 5))

# Angular velocity subplot
ax1 = plt.subplot(1, 2, 1)
ax1.plot(t_eval, omega1[:-1], label='ω1')
ax1.plot(t_eval, omega2[:-1], label='ω2')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angular Velocity (rad/s)')
ax1.set_title('Angular Velocities vs Time')
ax1.legend()
ax1.grid()

# Energy subplot
ax2 = plt.subplot(1, 2, 2)
ax2.plot(t_eval, energy[:-1])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Total Energy (J)')
ax2.set_title('Total Energy vs Time')
ax2.grid()

plt.tight_layout()
plt.show()