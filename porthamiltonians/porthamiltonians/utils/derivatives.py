
import numpy as np

__all__ = ['time_derivative']


def time_derivative(integrator, x_dot, x_start, x_end,
                    t_start, t_end, dt, u=None):
    """
    Computes the time derivative of x using the provided function *x_dot*.

    Parameters
    ----------
    integrator : str or bool
        If 'euler' or False, the time derivative at *x_start*, *t_start*
        is computed. If 'midpoint', *x_start*, *x_end*, *t_start*,
        *t_end* are used to compute the discretized implricit midpoint
        estimate of the derivative. If 'rk4', *x_start*, *t_start*, *dt*
        are used to compute the explicit Runge-Kutta 4 estimate.
        If 'srk4', *x_start*, *x_end*, *t_start*, *dt* are used to
        compute the symmetric runge kutta 4 estimate.
    x_dot : callable
        Callable taking three arguments, x, t and u, returning the time
        derivative at the provided points.
    x_start : (..., N) ndarray
    x_end : (..., N) ndarray
    t_start : number or (..., 1) ndarray
    t_end : number or (..., 1) ndarray
    dt : number
    u : (..., N) ndarray, default None
        Controlled input to provide to *x_dot*. Will only be used
        if *integrator* is 'srk4'.

    Returns
    -------
    (..., N) ndarray
        The estimated time derivatives.

    Raises
    ------
    ValueError
        If the integrator type is not recognized.

    """

    integrator = (integrator.lower() if isinstance(integrator, str)
                  else integrator)
    if integrator in (False, 'euler'):
        return _time_derivative_continuous(x_dot, x_start, t_start)
    elif integrator == 'midpoint':
        x_mid = (x_end + x_start) / 2
        t_mid = (t_end + t_start) / 2
        return _time_derivative_continuous(x_dot, x_mid, t_mid)
    elif integrator == 'rk4':
        return _discrete_time_derivative_rk4(x_dot, x_start, t_start, dt, u)
    elif integrator == 'srk4':
        return _discrete_time_derivative_srk4(x_dot, x_start, x_end,
                                              t_start, dt)
    else:
        raise ValueError(f'Unknown integrator {integrator}.')


def _time_derivative_continuous(x_dot, x, t=None):
    return x_dot(x, t)


def _discrete_time_derivative_rk4(x_dot, x1, t1, dt, u):
    k1 = x_dot(x1, t1, u)
    k2 = x_dot(x1+.5*dt*k1, t1+.5*dt, u)
    k3 = x_dot(x1+.5*dt*k2, t1+.5*dt, u)
    k4 = x_dot(x1+dt*k3, t1+dt, u)
    return 1/6*(k1+2*k2+2*k3+k4)


def _discrete_time_derivative_srk4(x_dot, x1, x2, t1, dt):
    xh = (x1+x2)/2
    z1 = (1/2+np.sqrt(3)/6)*x1 + (1/2-np.sqrt(3)/6)*x2
    z2 = (1/2-np.sqrt(3)/6)*x1 + (1/2+np.sqrt(3)/6)*x2
    tm = (t1 + (1/2-np.sqrt(3)/6)*dt)
    tp = (t1 + (1/2+np.sqrt(3)/6)*dt)
    z3 = xh - np.sqrt(3)/6*dt*x_dot(z2, tp)
    z4 = xh + np.sqrt(3)/6*dt*x_dot(z1, tm)
    return 1/2*(x_dot(z3, tm)+x_dot(z4, tp))
