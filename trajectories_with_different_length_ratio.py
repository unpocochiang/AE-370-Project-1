import numpy as np
import matplotlib.pyplot as plt
import eom
import num_method

# Constants
g = 9.81  # gravitational acceleration in m/s^2
l1 = 9.0  # length of pendulum 1 in meters
l2 = 1.0  # length of pendulum 2 in meters
m1 = 1.0  # mass of pendulum 1 in kg
m2 = 1.0  # mass of pendulum 2 in kg

l1_2 = 1.0  # length of pendulum 1 in meters
l2_2 = 15.0  # length of pendulum 15 in meters
m1_2 = 1.0  # mass of pendulum 1 in kg
m2_2 = 1.0  # mass of pendulum 2 in kg

# Initial conditions for two trajectories with small variations in initial angle
initial_conditions_1 = [np.pi / 2, np.pi / 2 + 0.01, 0, 0]  # Set 1: initial angles (90 degrees) and velocities (0)

# Time settings
t_max = 20  # maximum simulation time
dt = 0.01  # time step
num_steps = int(t_max / dt)

# Time array for plotting
t_eval = np.linspace(0, t_max, num_steps)

# Arrays to store the results
theta1_1, theta2_1, omega1_1, omega2_1 = [initial_conditions_1[0]], [initial_conditions_1[1]], [initial_conditions_1[2]], [initial_conditions_1[3]]
theta1_2, theta2_2, omega1_2, omega2_2 = [initial_conditions_1[0]], [initial_conditions_1[1]], [initial_conditions_1[2]], [initial_conditions_1[3]]

# Numerical integration using RK4 method
y1 = np.array(initial_conditions_1)
y2 = np.array(initial_conditions_1)

for t in t_eval[1:]:
    y1 = num_method.rk4_step(eom.double_pendulum, t, y1, dt, m1, m2, l1, l2)
    y2 = num_method.rk4_step(eom.double_pendulum, t, y2, dt, m1_2, m2_2, l1_2, l2_2)
    
    # Store the results
    theta1_1.append(y1[0])
    theta2_1.append(y1[1])
    omega1_1.append(y1[2])
    omega2_1.append(y1[3])
    
    theta1_2.append(y2[0])
    theta2_2.append(y2[1])
    omega1_2.append(y2[2])
    omega2_2.append(y2[3])

# Plot the results
plt.figure(figsize=(12, 6))

# Plot trajectories for both initial conditions
plt.subplot(2, 2, 1)
plt.plot(t_eval, theta1_1, label=r'$\theta_1$', color='b')
plt.plot(t_eval, theta2_1, label=r'$\theta_2$', color='g')
plt.title(r'Trajectories for lengths ($l_1$ = 9, $l_2$ = 1)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

# Plot phase space (theta1 vs omega1 and theta2 vs omega2)
plt.subplot(2, 2, 2)
plt.plot(theta1_1, omega1_1, label=r'Phase Space', color='b')
plt.plot(theta2_1, omega2_1, label=r'Phase Space', color='g')
plt.title('Phase Space')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()

# Plot trajectories for both initial conditions
plt.subplot(2, 2, 3)
plt.plot(t_eval, theta1_2, label=r'$\theta_1$', linestyle='dashed', color='b')
plt.plot(t_eval, theta2_2, label=r'$\theta_2$', linestyle='dashed', color='g')
plt.title(r'Trajectories for different lengths ($l_1$ = 15, $l_2$ = 1)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

# Plot phase space (theta1 vs omega1 and theta2 vs omega2)
plt.subplot(2, 2, 4)
plt.plot(theta1_2, omega1_2, label=r'Phase Space', linestyle='dashed', color='b')
plt.plot(theta2_2, omega2_2, label=r'Phase Space', linestyle='dashed', color='g')
plt.title('Phase Space')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()

plt.tight_layout()
plt.show()
