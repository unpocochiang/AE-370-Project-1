import numpy as np
import matplotlib.pyplot as plt

# Define the simple harmonic oscillator
omega = 2 * np.pi

def harmonic_oscillator(t, y):
    """ Defines the simple harmonic oscillator system. """
    x, v = y
    return np.array([v, -omega**2 * x])

# Exact solution for validation
def exact_solution(t, x0=1):
    """ Returns the exact solution x(t) for the harmonic oscillator. """
    return x0 * np.cos(omega * t)

# Implement RK4 solver
def RK4_solver(f, y0, t0, tf, h):
    """ Implements the Runge-Kutta 4th order method. """
    t_values = np.arange(t0, tf + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    
    for i in range(len(t_values) - 1):
        t, y = t_values[i], y_values[i]
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y_values[i + 1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t_values, y_values

# Run RK4 for different step sizes and compute errors
time_steps = [0.1, 0.05, 0.025, 0.0125]  # Reduce by factor of 2
errors = []

for h in time_steps:
    t_values, x_values = RK4_solver(harmonic_oscillator, np.array([1, 0]), 0, 1, h)
    exact_x = exact_solution(t_values)
    error = np.max(np.abs(x_values[:, 0] - exact_x))  # Max absolute error in position
    errors.append(error)

# Log-log plot to verify O(h^4) scaling
plt.figure(figsize=(8, 6))
plt.loglog(time_steps, errors, marker='o', label="RK4 Error")
plt.loglog(time_steps, [errors[0] * (h / time_steps[0])**4 for h in time_steps], 'r--', label=r"$O(h^4)$ Reference")
plt.xlabel("Time Step Size h")
plt.ylabel("Max Absolute Error")
plt.legend()
plt.title("RK4 Convergence Study: Error vs. Time Step")
plt.grid(True, which="both", linestyle="--")
plt.show()
