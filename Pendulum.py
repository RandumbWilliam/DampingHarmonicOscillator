#Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate

#Main class
class DampedPendulum:
    def __init__(self, init_state, l, m, g, b, origin):
        self.init_state = np.asarray(init_state, dtype='float')
        self.origin = np.asarray(origin, dtype='float')
        self.l = l
        self.m = m
        self.g = g
        self.b = b
        self.params = (l, m, g, b)
        self.t = np.linspace(0, 30, 200)
        self.time_elapsed = 0
        self.state = self.init_state * np.pi / 180.

    #Calculates the position of the mass
    def position(self):
        (l, m, g, b) = self.params
        
        x = np.cumsum([self.origin[0], l * np.sin(self.state[0])])
        y = np.cumsum([self.origin[1], -l * np.cos(self.state[0])])
        return (x, y)

    #Damping harmonic ODE
    def ode(self, x, t):
        (l, m, g, b) = self.params

        theta1 = x[0]
        theta2 = x[1]
        dtheta1_dt = theta2
        dtheta2_dt = -(b / m) * theta2 - (g / l) * np.sin(theta1)
        dtheta_dt = [dtheta1_dt, dtheta2_dt]
        return dtheta_dt

    #Solve
    def step(self, dt):
        self.state = integrate.odeint(self.ode, self.state, [0, dt])[1]
        self.time_elapsed += dt

    #Solution
    def theta(self):
        return integrate.odeint(self.ode, self.state, self.t)


# ------------------------------------------------------------
# set up initial state and global variables
pendulum = DampedPendulum([120, 0], 1.0, 1.0, 9.8, 0.5, (0, 0))
dt = 1. / 30  # 30 fps

# ------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
xlim = (-2 - pendulum.l, 2 + pendulum.l)
klim = (0, 30)

ylim = (-2 - pendulum.l, 2 + pendulum. l)
jlim = (-2 - max(pendulum.theta()[:,1]), 2 + max(pendulum.theta()[:,1]))

ax1 = fig.add_subplot(121, aspect=1, xlim=xlim, ylim=ylim)
ax1.grid()

line, = ax1.plot([], [], 'o-', lw=2, color='r')
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

ax2 = fig.add_subplot(122, xlim=klim, ylim=jlim)
ax2.grid()
ax2.plot(pendulum.t, pendulum.theta()[:,0], label=r'$\frac{d\theta_1}{dt}=\theta_2$')
ax2.plot(pendulum.t, pendulum.theta()[:,1], label=r'$\frac{d\theta_2}{dt}=-\frac{b}{m}\theta_2-\frac{g}{L}sin\theta_1$')
ax2.legend(loc='best')

def init():
    """initialize animation"""
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    """perform animation step"""
    global pendulum, dt
    pendulum.step(dt)

    line.set_data(*pendulum.position())
    time_text.set_text('time = %.1f' % pendulum.time_elapsed)
    return line, time_text


# choose the interval based on dt and the time to animate one step
from time import time

t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=300,
                              interval=interval, blit=True, init_func=init)

plt.show()
