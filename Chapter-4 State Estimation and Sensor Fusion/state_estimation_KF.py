import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, Matrix
from sympy.interactive import printing

# the initial state of pedestrian
# state:[Px, Py, Vx, Vy]
x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T

# the initial conv matrix of pedestrian
P = np.diag([1000.0, 1000.0, 1000.0, 1000.0])

dt = 0.1    # Time step between Filter Steps
# F: A， State transition matrix
F = np.matrix([[1.0, 0.0, dt, 0.0],
               [0.0, 1.0, 0.0, dt],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])

# H: C, observation measurement matrix
H = np.matrix([[0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])

ra = 0.01
# the cov matrix of measurement noise
R = np.matrix([[ra, 0.0],
              [0.0, ra]])

sv = 0.3
G = np.matrix([[0.5 * dt**2],
               [0.5 * dt**2],
               [dt],
               [dt]])
# the cov matrix of process noise
Q = G * G.T * sv**2

printing.init_printing()
dts = Symbol('dt')
Qs = Matrix([[0.5 * dts**2], [0.5 * dts**2], [dts], [dts]])

I = np.eye(4)
print(I, I.shape)

# generate measurement data
m = 200     # Measurements
vx = 20     # the mean of  X
vy = 10     # the mean of  Y

mx = np.array(vx + np.random.rand(m))
my = np.array(vy + np.random.rand(m))
measurements = np.vstack((mx, my))
print(measurements.shape)
print('Standard Deviation of Acceleration Measurement=%.2f' % np.std(mx))
print('You assumed %.2f in R' % R[0, 0])

fig = plt.figure(figsize=(16, 5))
plt.step(range(m), mx, label='$\dot x$')
plt.step(range(m), my, label='$\dot y$')
plt.ylabel(r'Velocity $m/s$')
plt.title('Measurements')
plt.legend(loc='best', prop={'size':18})

# Process data
xt = []
yt = []
dxt = []
dyt = []
Zx = []
Zy = []
Px = []
Py = []
Pdx = []
Pdy = []
Rdx = []
Rdy =[]
Kx = []
Ky = []
Kdx = []
Kdy =[]


def savestates(x, Z, P, R, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))

    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))

    Px.append(float(P[0, 0]))
    Py.append(float(P[1, 1]))
    Pdx.append(float(P[2, 2]))
    Pdy.append(float(P[3, 3]))

    Rdx.append(float(R[0, 0]))
    Rdy.append(float(R[1, 1]))

    Kx.append(float(K[0, 0]))
    Ky.append(float(K[1, 0]))
    Kdx.append(float(K[2, 0]))
    Kdy.append(float(K[3, 0]))


# Kalman Filter
for n in range(len(measurements[0])):

    # Time Update (Prediction)
    # ===========================
    # Project the state ahead
    x = F * x

    # Project the error covariance ahead
    P = F * P * F.T + Q

    # Measurement Update (Correction)
    # ==============================
    # Compute the Kalman Gain
    S = H * P * H.T + R
    K = (P * H.T) * np.linalg.pinv(S)

    # Update the estimate via z
    Z = measurements[:, n].reshape(2, 1)
    y = Z - (H * x)     # Innovation or Residual
    x = x + (K * y)

    # Update the error covariance
    P = (I - (K * H)) * P

    # Save states (for Plotting)
    savestates(x, Z, P, R, K)


def plot_x():
    fig = plt.figure(figsize=(16,9))
    plt.step(range(len(measurements[0])), dxt, label='$ estimateVx $')
    plt.step(range(len(measurements[0])), dyt, label='$ estimateVy $')

    plt.step(range(len(measurements[0])), measurements[0], label='$ measurementVx $')
    plt.step(range(len(measurements[0])), measurements[1], label='$ measurementVy $')

    plt.axhline(vx, color='#999999', label='$ trueVx $')
    plt.axhline(vy, color='#999999', label='$ trueVy $')

    plt.xlabel('Filter Step')
    plt.title('Estimate (Elements from State Vector $ x $)')
    plt.legend(loc='best', prop={'size':11})
    plt.ylim([0, 30])
    plt.ylabel('Velocity')


def plot_xy():
    fig = plt.figure(figsize=(16,16))
    plt.scatter(xt, yt, s=20, label='State', c='k')
    plt.scatter(xt[0], yt[0], s=100, label='Start', c='g')
    plt.scatter(xt[-1], yt[-1], s=100, label='Goal', c='r')

    plt.xlabel('X')
    plt.xlabel('Y')
    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')


plot_x()
plot_xy()
plt.show()






