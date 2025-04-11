import num_method
import eom
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravitational acceleration in m/s^2
l1 = 1.0  # length of pendulum 1 in meters
l2 = 1.0  # length of pendulum 2 in meters
m1 = 1.0  # mass of pendulum 1 in kg
m2 = 1.0  # mass of pendulum 2 in kg

# Initial conditions for two cases: 90° vs. 15°
initial_conditions_90 = [np.pi / 2, np.pi / 2, 0, 0]  # 90° initial angle
initial_conditions_15 = [np.pi / 12, np.pi / 12, 0, 0]  # 15° initial angle

# Time settings
t_max = 40  # maximum simulation time
dt = 0.01  # time step
num_steps = int(t_max / dt)

# Time array for plotting
t_eval = np.linspace(0, t_max, num_steps)

# Arrays to store the results
theta1_90, theta2_90 = [initial_conditions_90[0]], [initial_conditions_90[1]]
theta1_15, theta2_15 = [initial_conditions_15[0]], [initial_conditions_15[1]]

# Numerical integration using RK4 method
y90 = np.array(initial_conditions_90)
y15 = np.array(initial_conditions_15)

for t in t_eval[1:]:
    y90 = num_method.rk4_step(eom.double_pendulum, t, y90, dt, m1, m2, l1, l2)
    y15 = num_method.rk4_step(eom.double_pendulum, t, y15, dt, m1, m2, l1, l2)
    
    # Store the results
    theta1_90.append(y90[0])
    theta2_90.append(y90[1])
    
    theta1_15.append(y15[0])
    theta2_15.append(y15[1])

# Plot the results
plt.figure(figsize=(10, 5))

# Plot trajectories for both initial conditions
plt.plot(t_eval, theta1_90, label=r'$\theta_1$ (90°)', color='b')
plt.plot(t_eval, theta2_90, label=r'$\theta_2$ (90°)', linestyle='dashed', color='b')

plt.plot(t_eval, theta1_15, label=r'$\theta_1$ (15°)', color='r')
plt.plot(t_eval, theta2_15, label=r'$\theta_2$ (15°)', linestyle='dashed', color='r')

plt.title('Angle vs. Time for 90° and 15° Initial Conditions')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid()

plt.show()
