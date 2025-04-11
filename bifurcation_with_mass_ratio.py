import numpy as np
import matplotlib.pyplot as plt
import num_method
import eom

# Parameters
g = 9.81
l1 = 1.0
l2 = 1.0

# Initial conditions and time setup
y0 = [np.pi/2, 0, np.pi/2 + 0.01, 0]
t_max = 50
dt = 0.001
t_eval = np.arange(0, t_max, dt)

# Sweep through m2 (θ2 peaks) and m1 (θ1 peaks)
m1_fixed = 1.0
m2_fixed = 1.0
m2_values = np.linspace(0.1, 100.0, 100)
m1_values = np.linspace(0.1, 100.0, 100)

theta2_peaks = []
theta1_peaks = []

# θ2 peaks (sweep m2, fixed m1)
for m2 in m2_values:
    y = np.array(y0)
    θ2_vals = []
    for t in t_eval:
        θ2_vals.append(y[2])
        y = num_method.rk4_step(eom.double_pendulum, t, y, dt, m1_fixed, m2, l1, l2)
    theta2 = np.array(θ2_vals[int(len(θ2_vals)/2):])
    second_diff = np.diff(np.sign(np.diff(theta2)))
    peak_indices = np.where(second_diff < 0)[0] + 1
    peaks = theta2[peak_indices]
    theta2_peaks.append(peaks[-10:] if len(peaks) >= 10 else [np.nan]*10)

# θ1 peaks (sweep m1, fixed m2)
for m1 in m1_values:
    y = np.array(y0)
    θ1_vals = []
    for t in t_eval:
        θ1_vals.append(y[2])
        y = num_method.rk4_step(eom.double_pendulum, t, y, dt, m1, m2_fixed, l1, l2)
    theta1 = np.array(θ1_vals[int(len(θ1_vals)/2):])
    second_diff = np.diff(np.sign(np.diff(theta1)))
    peak_indices = np.where(second_diff < 0)[0] + 1
    peaks = theta1[peak_indices]
    theta1_peaks.append(peaks[-10:] if len(peaks) >= 10 else [np.nan]*10)

# Mass ratios
mass_ratio_θ2 = m1_fixed / m2_values
mass_ratio_θ1 = m1_values / m2_fixed

# Plot overlay
plt.figure(figsize=(10, 6))
for i in range(10):
    plt.plot(mass_ratio_θ2, [p[i] if len(p) > i else np.nan for p in theta2_peaks], 'b.', markersize=1)
    plt.plot(mass_ratio_θ1, [p[i] if len(p) > i else np.nan for p in theta1_peaks], 'b.', markersize=1)
plt.ylim(-1.2, 7.2)
plt.xlabel('Mass Ratio (m₁/m₂)')
plt.ylabel('θ₂ Peak Angle (radians)')
plt.title('Bifurcation Diagram of θ₂ vs m₁/m₂ (RK4)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
