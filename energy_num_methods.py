import num_method
import eom
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.81  # gravitational acceleration in m/s^2
l1 = 1.0  # length of pendulum 1 in meters
l2 = 1.0  # length of pendulum 2 in meters
m1 = 1.0  # mass of pendulum 1 in kg
m2 = 1.0  # mass of pendulum 2 in kg=

# Initial conditions
initial_conditions = [np.pi / 2, np.pi / 2, 0, 0]  # Set 1: initial angles (90 degrees) and velocities (0)
initial_conditions_1 = [np.pi / 12, np.pi / 12, 0, 0]  # Set 2: initial angles (15 degrees) and velocities (0)

# Time settings
t_max = 100  # maximum simulation time
dt = 0.01  # time step
num_steps = int(t_max / dt)

# Time array for plotting
t_eval = np.linspace(0, t_max, num_steps)

# Arrays to store the results for energy
energy_euler, energy_rk4, energy_symplectic = [], [], []
energy_euler_1, energy_rk4_1, energy_symplectic_1 = [], [], []

# Initial conditions for the numerical methods
y_euler = np.array(initial_conditions)
y_rk4 = np.array(initial_conditions)
y_symplectic = np.array(initial_conditions)

# Initial conditions for the numerical methods
y_euler_1 = np.array(initial_conditions_1)
y_rk4_1 = np.array(initial_conditions_1)
y_symplectic_1 = np.array(initial_conditions_1)

# Calculate energy function (kinetic + potential)
def calculate_energy(y):
    theta1, theta2, omega1, omega2 = y
    T = 0.5 * m1 * l1**2 * omega1**2 + 0.5 * m2 * l1**2 * omega1**2 + 0.5 * m2 * l2**2 * omega2**2 + m2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    V = -m1 * g * l1 * np.cos(theta1) - m2 * g * (l1 * np.cos(theta1) + l2 * np.cos(theta2))
    return T + V

# Simulate the system for different methods
for t in t_eval:
    # Euler method
    energy_euler.append(calculate_energy(y_euler))
    y_euler = num_method.euler_step(eom.double_pendulum, t, y_euler, dt, m1, m2, l1, l2)

    # RK4 method
    energy_rk4.append(calculate_energy(y_rk4))
    y_rk4 = num_method.rk4_step(eom.double_pendulum, t, y_rk4, dt, m1, m2, l1, l2)

    # Symplectic method (fixed)
    energy_symplectic.append(calculate_energy(y_symplectic))
    y_symplectic = num_method.symplectic_step(eom.double_pendulum, t, y_symplectic, dt, m1, m2, l1, l2)

# Simulate the system for different methods
for t in t_eval:
    # Euler method
    energy_euler_1.append(calculate_energy(y_euler_1))
    y_euler_1 = num_method.euler_step(eom.double_pendulum, t, y_euler_1, dt, m1, m2, l1, l2)

    # RK4 method
    energy_rk4_1.append(calculate_energy(y_rk4_1))
    y_rk4_1 = num_method.rk4_step(eom.double_pendulum, t, y_rk4_1, dt, m1, m2, l1, l2)

    # Symplectic method (fixed)
    energy_symplectic_1.append(calculate_energy(y_symplectic_1))
    y_symplectic_1 = num_method.symplectic_step(eom.double_pendulum, t, y_symplectic_1, dt, m1, m2, l1, l2)

# Plot the energy for each method
plt.figure(figsize=(10, 6))

plt.plot(t_eval, energy_euler, label='Euler Method', linestyle='-', color='r')
plt.plot(t_eval, energy_rk4, label='RK4 Method', linestyle='-', color='g')
plt.plot(t_eval, energy_symplectic, label='Symplectic Method', linestyle='-', color='b')

plt.title('Energy Conservation for Different Numerical Methods (Initial Angle 90 Degrees)')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (Joules)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(10, 6))

plt.plot(t_eval, energy_euler_1, label='Euler Method', linestyle='-', color='r')
plt.plot(t_eval, energy_rk4_1, label='RK4 Method', linestyle='-', color='g')
plt.plot(t_eval, energy_symplectic_1, label='Symplectic Method', linestyle='-', color='b')

plt.title('Energy Conservation for Different Numerical Methods (Initial Angle 15 Degrees)')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy (Joules)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
