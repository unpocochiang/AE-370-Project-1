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

# Initial conditions for two trajectories with small variations in initial angle
initial_conditions_1 = [np.pi / 2, np.pi / 2, 0, 0]  # Set 1: initial angles (90 degrees) and velocities (0)
initial_conditions_2 = [np.pi / 2 + 0.1, np.pi / 2, 0, 0]  # Set 2: initial angle for pendulum 1 is slightly different

# Time settings
t_max = 20  # maximum simulation time
dt = 0.01  # time step
num_steps = int(t_max / dt)

# Time array for plotting
t_eval = np.linspace(0, t_max, num_steps)

# Arrays to store the results
theta1_1, theta2_1, omega1_1, omega2_1 = [initial_conditions_1[0]], [initial_conditions_1[1]], [initial_conditions_1[2]], [initial_conditions_1[3]]
theta1_2, theta2_2, omega1_2, omega2_2 = [initial_conditions_2[0]], [initial_conditions_2[1]], [initial_conditions_2[2]], [initial_conditions_2[3]]

# Arrays to store the separation for Lyapunov exponent calculation
separation = [1e-5]  # Increased initial separation
separation_2 = [0]

# Numerical integration using RK4 method
y1 = np.array(initial_conditions_1)
y2 = np.array(initial_conditions_2)

for t in t_eval[1:]:
    y1 = num_method.rk4_step(eom.double_pendulum, t, y1, dt, m1, m2, l1, l2)
    y2 = num_method.rk4_step(eom.double_pendulum, t, y2, dt, m1, m2, l1, l2)
    
    # Store the results
    theta1_1.append(y1[0])
    theta2_1.append(y1[1])
    omega1_1.append(y1[2])
    omega2_1.append(y1[3])
    
    theta1_2.append(y2[0])
    theta2_2.append(y2[1])
    omega1_2.append(y2[2])
    omega2_2.append(y2[3])
    
    # Calculate the separation between the two trajectories
    sep = np.sqrt((y1[0] - y2[0])**2 + (y1[1] - y2[1])**2 + (y1[2] - y2[2])**2 + (y1[3] - y2[3])**2)
    separation.append(sep)
    separation_2.append(sep)

# Calculate the average Lyapunov exponent as the rate of divergence
separation_2[0] = max(separation_2[1], 1e-10)  # Ensure the initial separation is non-zero
separation = np.array(separation)
separation_2 = np.array(separation_2)
lyapunov_exponent = np.log(separation / separation[0]) / t_eval  # Approximation of the exponent
lyapunov_exponent_2 = np.log(separation_2 / separation_2[0]) / t_eval  # Approximation of the exponent
avg_lyapunov_exponent = np.mean(lyapunov_exponent[int(len(lyapunov_exponent)/4):])  # Average from halfway to end for stability

# Plot the results
plt.figure(figsize=(12, 6))

# Plot trajectories for both initial conditions
plt.subplot(1, 2, 1)
plt.plot(t_eval, theta1_1, label=r'$\theta_1$ (Set 1)', color='b')
plt.plot(t_eval, theta2_1, label=r'$\theta_2$ (Set 1)', color='g')
plt.plot(t_eval, theta1_2, label=r'$\theta_1$ (Set 2)', linestyle='dashed', color='b')
plt.plot(t_eval, theta2_2, label=r'$\theta_2$ (Set 2)', linestyle='dashed', color='g')
plt.title('Trajectories for Different Initial Conditions')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()

# Plot phase space (theta1 vs omega1 and theta2 vs omega2)
plt.subplot(1, 2, 2)
plt.plot(theta1_1, omega1_1, label=r'Phase Space (Set 1)', color='b')
plt.plot(theta2_1, omega2_1, label=r'Phase Space (Set 1)', color='g')
plt.plot(theta1_2, omega1_2, label=r'Phase Space (Set 2)', linestyle='dashed', color='b')
plt.plot(theta2_2, omega2_2, label=r'Phase Space (Set 2)', linestyle='dashed', color='g')
plt.title('Phase Space')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()

# Plot Lyapunov Exponent
plt.figure(figsize=(8, 6))
plt.plot(t_eval, lyapunov_exponent, label=r'$\lambda$ (Lyapunov Exponent)', color='r')
plt.title(f'Lyapunov Exponent Over Time (Average: {avg_lyapunov_exponent:.2f})')
plt.xlabel('Time (s)')
plt.ylabel('Lyapunov Exponent')
plt.legend()

# Plot Lyapunov Exponent
plt.figure(figsize=(8, 6))
plt.plot(t_eval, lyapunov_exponent_2, label=r'$\lambda$ (Lyapunov Exponent)', color='r')
plt.title('Lyapunov Exponent Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Lyapunov Exponent')
plt.legend()

plt.tight_layout()
plt.show()