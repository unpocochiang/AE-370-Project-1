import numpy as np

# Equations of motion for the double pendulum (derived from Lagrangian mechanics)
def double_pendulum(t, y, m1, m2, l1, l2):
    theta1, theta2, omega1, omega2 = y

    dtheta1_dt = omega1
    dtheta2_dt = omega2
    g = 9.81
    domega1_dt = m2*(-g*l2*m1*np.sin(theta1) - g*l2*m2*np.sin(theta1) 
                     + g*np.sin(theta2)*np.cos(theta1 - theta2) - l1*omega1**2*np.sin(2*theta1 - 2*theta2)/2 
                     - l2*omega2**2*np.sin(theta1 - theta2))/(l1*(m1 - m2*np.cos(theta1 - theta2)**2 + m2))
    domega2_dt = (g*l2*m1*m2*np.sin(theta1)*np.cos(theta1 - theta2) + g*l2*m2**2*np.sin(theta1)*np.cos(theta1 - theta2) 
                  - g*m1*np.sin(theta2) - g*m2*np.sin(theta2) + l1*m1*omega1**2*np.sin(theta1 - theta2) 
                  + l1*m2*omega1**2*np.sin(theta1 - theta2) 
                  + l2*m2*omega2**2*np.sin(2*theta1 - 2*theta2)/2)/(l2*(m1 - m2*np.cos(theta1 - theta2)**2 + m2))
    
    return [dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt]

# Double pendulum dynamics with damping
def double_pendulum_with_damping(t, y, m1, m2, l1, l2, b):
    theta1, theta2, omega1, omega2 = y

    dtheta1_dt = omega1
    dtheta2_dt = omega2
    g = 9.81
    domega1_dt = (m2*(-g*l2*m1*np.sin(theta1) - g*l2*m2*np.sin(theta1) + 
                 g*np.sin(theta2)*np.cos(theta1 - theta2) - 
                 l1*omega1**2*np.sin(2*theta1 - 2*theta2)/2 - 
                 l2*omega2**2*np.sin(theta1 - theta2)) / 
                 (l1*(m1 - m2*np.cos(theta1 - theta2)**2 + m2))) - b * omega1
    
    domega2_dt = ((g*l2*m1*m2*np.sin(theta1)*np.cos(theta1 - theta2) + 
                  g*l2*m2**2*np.sin(theta1)*np.cos(theta1 - theta2) - 
                  g*m1*np.sin(theta2) - g*m2*np.sin(theta2) + 
                  l1*m1*omega1**2*np.sin(theta1 - theta2) + 
                  l1*m2*omega1**2*np.sin(theta1 - theta2) + 
                  l2*m2*omega2**2*np.sin(2*theta1 - 2*theta2)/2) / 
                  (l2*(m1 - m2*np.cos(theta1 - theta2)**2 + m2))) - b * omega2
    
    return [dtheta1_dt, dtheta2_dt, domega1_dt, domega2_dt]
