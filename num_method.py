import numpy as np
import matplotlib.pyplot as plt

# Implementing the RK4 method for numerical integration
def rk4_step(f, t, y, dt, *args):
    k1 = dt * np.array(f(t, y, *args))
    k2 = dt * np.array(f(t + 0.5 * dt, y + 0.5 * k1, *args))
    k3 = dt * np.array(f(t + 0.5 * dt, y + 0.5 * k2, *args))
    k4 = dt * np.array(f(t + dt, y + k3, *args))
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def euler_step(f, t, y, dt, *args):
    return y + dt * np.array(f(t, y, *args))

# Corrected Symplectic Euler method
def symplectic_step(f, t, y, dt, *args):
    theta1, theta2, omega1, omega2 = y
    
    # Update angular velocities first
    domega1_dt, domega2_dt = f(t, y, *args)[2:]
    omega1_new = omega1 + dt * domega1_dt
    omega2_new = omega2 + dt * domega2_dt

    # Update angles using the new velocities
    theta1_new = theta1 + dt * omega1_new
    theta2_new = theta2 + dt * omega2_new
    
    return np.array([theta1_new, theta2_new, omega1_new, omega2_new])