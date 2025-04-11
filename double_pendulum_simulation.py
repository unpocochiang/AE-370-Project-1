import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === User Inputs ===
print("Welcome to the Double Pendulum Simulator!")
try:
    m1 = float(input("Mass of pendulum 1 (kg) [default: 1.0]: ") or 1.0)
    m2 = float(input("Mass of pendulum 2 (kg) [default: 1.0]: ") or 1.0)
    l1 = float(input("Length of rod 1 (m) [default: 1.0]: ") or 1.0)
    l2 = float(input("Length of rod 2 (m) [default: 1.0]: ") or 1.0)
    theta1_0 = float(input("Initial angle θ1 (deg) [default: 90]: ") or 90)
    theta2_0 = float(input("Initial angle θ2 (deg) [default: 90]: ") or 90)
    g = float(input("Gravity [default: 9.81]: ") or 9.81)
    t_max = float(input("Simulation time (seconds) [default: 40]: ") or 40)
except ValueError:
    print("Invalid input detected. Exiting...")
    exit()

# Convert degrees to radians
theta1_0 = np.radians(theta1_0)
theta2_0 = np.radians(theta2_0)

# Constants
y0 = [theta1_0, theta2_0, 0.0, 0.0]  # [theta1, theta2, omega1, omega2]
dt = 0.001
t_eval = np.arange(0, t_max, dt)

# === RK4 Integrator ===
def rk4_step(f, t, y, dt, *args):
    k1 = dt * np.array(f(t, y, *args))
    k2 = dt * np.array(f(t + 0.5 * dt, y + 0.5 * k1, *args))
    k3 = dt * np.array(f(t + 0.5 * dt, y + 0.5 * k2, *args))
    k4 = dt * np.array(f(t + dt, y + k3, *args))
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# === Double Pendulum Dynamics ===
def double_pendulum(t, y, m1, m2, l1, l2):
    theta1, theta2, omega1, omega2 = y

    delta = theta2 - theta1
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
    den2 = (l2 / l1) * den1

    domega1_dt = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                  m2 * g * np.sin(theta2) * np.cos(delta) +
                  m2 * l2 * omega2**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1

    domega2_dt = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                  (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                  (m1 + m2) * l1 * omega1**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta2)) / den2

    return [omega1, omega2, domega1_dt, domega2_dt]

# === Integrate ===
states = [y0]
y = np.array(y0)
for t in t_eval:
    y = rk4_step(double_pendulum, t, y, dt, m1, m2, l1, l2)
    states.append(y)

states = np.array(states)

# === Compute Positions ===
theta1 = states[:, 0]
theta2 = states[:, 1]
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# === Set up Animation ===
fig, ax = plt.subplots(figsize=(6, 6))
length = max(l1 + l2, 2)
ax.set_xlim(-length * 1.1, length * 1.1)
ax.set_ylim(-length * 1.1, length * 1.1)
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], 'r-', alpha=0.4, linewidth=1)
trace_x, trace_y = [], []

def init():
    line.set_data([], [])
    trace.set_data([], [])
    return line, trace

def update(frame):
    this_x = [0, x1[frame], x2[frame]]
    this_y = [0, y1[frame], y2[frame]]
    line.set_data(this_x, this_y)

    trace_x.append(x2[frame])
    trace_y.append(y2[frame])
    trace.set_data(trace_x, trace_y)
    
    return line, trace

skip = 10
ani = FuncAnimation(fig, update, frames=range(0, len(t_eval), skip), init_func=init, blit=True, interval=10)
plt.title("Double Pendulum Animation (RK4)")
plt.show()
