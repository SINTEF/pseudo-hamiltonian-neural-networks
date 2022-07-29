
import numpy as np


def time_derivative(integrator, x_dot, x_start, x_end, t_start, t_end, dt, u=None):
    integrator = integrator.lower() if isinstance(integrator, str) else integrator
    if integrator in (False, 'euler'):
        return time_derivative_continuous(x_dot, x_start, t_start)
    elif integrator == 'midpoint':
        x_mid = (x_end + x_start) / 2
        t_mid = (t_end + t_start) / 2
        return time_derivative_continuous(x_dot, x_mid, t_mid)
    elif integrator == 'rk4':
        return discrete_time_derivative_rk4(x_dot, x_start, t_start, dt, u)
    elif integrator == 'srk4':
        return discrete_time_derivative_srk4(x_dot, x_start, x_end, t_start, dt)
    else:
        raise ValueError(f'Unknown integrator {integrator}.')


def time_derivative_continuous(x_dot, x, t=None):
    return x_dot(x=x, t=t)


def discrete_time_derivative_rk4(x_dot, x1, t1, dt, u):
    k1 = x_dot(x1, t1, u)
    k2 = x_dot(x1+.5*dt*k1, t1+.5*dt, u)
    k3 = x_dot(x1+.5*dt*k2, t1+.5*dt, u)
    k4 = x_dot(x1+dt*k3, t1+dt, u)
    return 1/6*(k1+2*k2+2*k3+k4)


def discrete_time_derivative_srk4(x_dot, x1, x2, t1, dt):
    xh = (x1+x2)/2
    z1 = (1/2+np.sqrt(3)/6)*x1 + (1/2-np.sqrt(3)/6)*x2
    z2 = (1/2-np.sqrt(3)/6)*x1 + (1/2+np.sqrt(3)/6)*x2
    tm = (t1 + (1/2-np.sqrt(3)/6)*dt)
    tp = (t1 + (1/2+np.sqrt(3)/6)*dt)
    z3 = xh - np.sqrt(3)/6*dt*x_dot(z2, tp)
    z4 = xh + np.sqrt(3)/6*dt*x_dot(z1, tm)
    return 1/2*(x_dot(z3, tm)+x_dot(z4, tp))
