import numpy as np
import matplotlib.pyplot as plt

def euler_step(f, y, t, h):
    return y + h * f(t, y)

def rk2_step(f, y, t, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    return y + h * k2

def rk4_step(f, y, t, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

def f(t, y):
    return -y  # Simple test equation dy/dt = -y (exponential decay)

def compute_error(method, h_values, y_exact, t_final=1.0):
    errors = []
    for h in h_values:
        y = 1.0  # Initial condition
        t = 0.0
        while t < t_final:
            y = method(f, y, t, h)
            t += h
        errors.append(abs(y - y_exact))
    return np.array(errors)

h_values = np.logspace(-3, -1, 10)
y_exact = np.exp(-1)  # Exact solution at t=1

errors_euler = compute_error(euler_step, h_values, y_exact)
errors_rk2 = compute_error(rk2_step, h_values, y_exact)
errors_rk4 = compute_error(rk4_step, h_values, y_exact)

plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors_euler, 'o-', label='Euler (O(h))')
plt.loglog(h_values, errors_rk2, 's-', label='RK2 (O(h^2))')
plt.loglog(h_values, errors_rk4, 'd-', label='RK4 (O(h^4))')
plt.loglog(h_values, h_values, 'k--', label='O(h) Reference')
plt.loglog(h_values, h_values**2, 'k-.', label='O(h^2) Reference')
plt.loglog(h_values, h_values**4, 'k:', label='O(h^4) Reference')
plt.xlabel('Step Size h')
plt.ylabel('Error at t=1')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Error Convergence of Euler, RK2, and RK4')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def euler_step(f, y, t, h):
    return y + h * f(t, y)

def rk2_step(f, y, t, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    return y + h * k2

def rk4_step(f, y, t, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h / 2 * k1)
    k3 = f(t + h / 2, y + h / 2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h / 6 * (k1 + 2*k2 + 2*k3 + k4)

def f(t, y):
    return -y  # Simple test equation dy/dt = -y (exponential decay)

def compute_max_error(method, h_values, y_exact_func, t_final=1.0):
    errors = []
    for h in h_values:
        y = 1.0  # Initial condition
        t = 0.0
        max_error = 0.0
        while t < t_final:
            y = method(f, y, t, h)
            t += h
            exact = y_exact_func(t)
            max_error = max(max_error, abs(y - exact))
        errors.append(max_error)
    return np.array(errors)

h_values = np.logspace(-3, -1, 10)
y_exact_func = lambda t: np.exp(-t)  # Exact solution y = e^(-t)

errors_euler = compute_max_error(euler_step, h_values, y_exact_func)
errors_rk2 = compute_max_error(rk2_step, h_values, y_exact_func)
errors_rk4 = compute_max_error(rk4_step, h_values, y_exact_func)

plt.figure(figsize=(8, 6))
plt.loglog(h_values, errors_euler, 'o-', label='Euler (O(h))')
plt.loglog(h_values, errors_rk2, 's-', label='RK2 (O(h^2))')
plt.loglog(h_values, errors_rk4, 'd-', label='RK4 (O(h^4))')
plt.loglog(h_values, h_values, 'k--', label='O(h) Reference')
plt.loglog(h_values, h_values**2, 'k-.', label='O(h^2) Reference')
plt.loglog(h_values, h_values**4, 'k:', label='O(h^4) Reference')
plt.xlabel('Step Size h')
plt.ylabel('Max Absolute Error')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.title('Error Convergence of Euler, RK2, and RK4')
plt.show()