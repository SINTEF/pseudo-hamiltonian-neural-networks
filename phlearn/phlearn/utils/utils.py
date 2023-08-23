
import torch
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import imageio
from IPython.display import display, Video, Image

__all__ = ['to_tensor', 'midpoint_method', 'create_video']

def to_tensor(x, ttype=torch.float32):
    """
    Converts the input to a torch tensor if the input is not None.

    Parameters
    ----------
    x : listlike or None
    ttype : torch type, default torch.float32

    Returns
    -------
    torch.tensor or None
        Return converted list/array/tensor unless *x* is None,
        in which case it returns None.

    """
    if x is None:
        return x
    elif not isinstance(x, torch.Tensor):
        return torch.tensor(x, dtype=ttype)
    else:
        return x


def midpoint_method(u, un, t, f, Df, dt, M, tol=1e-12, max_iter=5):
    """
    Integrates one step of the ODE system u_t = f,
    from u to un, with the implicit midpoint method.
    Uses Newton's method to find un.
    
    Parameters
    ----------
    u : ndarray, shape (M,)
        Initial state of the ODE system.
    un : ndarray, shape (M,)
        Initial guess on the state of the ODE system after one time step.
    t : float
        Time at the initial state.
    f : callable
        Function that evaluates the right-hand side of the ODE system,
        given the state u and the time t.
        It should return an array_like of shape (M,).
    Df : callable
        Function that evaluates the Jacobian matrix of f,
        given the state u and the time t.
        It should return an array_like of shape (M, M).
    dt : float
        Time step size.
    M : int
        Number of equations in the ODE system.
    tol : float, optional
        Tolerance for the Newton iteration. The iteration stops when the
        Euclidean norm of the residual is less than `tol`. Default is 1e-12.
    max_iter : int, optional
        Maximum number of iterations for the Newton iteration. If the
        iteration does not converge after `max_iter` iterations, it stops.
        Default is 5.


    Returns
    -------
    un : array_like, shape (M,)
        Final state of the ODE system after one time step.
    """

    I = np.eye(M)
    F = lambda u_hat: 1/dt*(u_hat-u) - f((u+u_hat)/2, t+.5*dt)
    J = lambda u_hat: 1/dt*I - 1/2*Df((u+u_hat)/2, t+.5*dt)
    err = la.norm(F(un))
    it = 0
    while err > tol:
        un = un - la.solve(J(un),F(un))
        err = la.norm(F(un))
        it += 1
        if it > max_iter:
            break
    return un


def create_video(arrays, labels, x_axis=None, file_name='animation.mp4', fps=10, dpi=100, output_format='MP4'):
    """
    Creates an MP4 video or GIF showing the evaluation of a system over time, given by data in an
    array where the time scale is along the first dimension.

    Parameters
    ----------
    arrays : list of ndarray
        List of numpy arrays containing the data to be plotted.
    labels : list of str
        List of labels for the data series, used in the legend of the plot, corresponding to the
        list of arrays.
    x_axis : ndarray, optional
        x-axis values. If provided, data will be plotted against these x-axis values. If None,
        data will be plotted against the array indices.
    file_name : str, optional
        Name of the output animation file. Defaults to 'animation.mp4'.
    fps : int, optional
        Frames per second for the animation. Defaults to 10.
    dpi : int, optional
        Dots per inch for the plot's resolution. Defaults to 100.
    output_format : str, optional
        Output format for the animation. Can be 'MP4' or 'GIF'.
        Defaults to 'MP4'.

    Returns
    -------
    Video or None
        Returns a Video object if the output_format is 'MP4'. If output_format is 'GIF',
        the GIF is displayed and None is returned.
    """

    fig, ax = plt.subplots(figsize=(7.04, 4), dpi=dpi)
    colors = [(0,0,0),(0,0.4,1),(1,0.7,0.3),(0.2,0.7,0.2),(0.8,0,0.2),(0.5,0.3,.9)][:len(arrays)]
    min_value = max(min(np.min(data) for data in arrays), -2)
    max_value = min(max(np.max(data) for data in arrays), 2*np.max(arrays[0]))
    ax.set_ylim(min_value, max_value)

    with imageio.get_writer(file_name, mode='I', fps=fps) as writer:
        for frame in range(arrays[0].shape[0]):
            lines = []
            for (data, color, label) in zip(arrays, colors, labels):
                if x_axis is not None:
                    line, = ax.plot(x_axis, data[frame, :], color=color, label=label)
                else:
                    line, = ax.plot(data[frame, :], color=color, label=label)
                lines.append(line)
            ax.legend(loc='upper right')
            ax.set_xlabel('$x$', fontsize=12)
            ax.set_ylabel('$u$', fontsize=12)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
            ax.clear()
            ax.set_ylim(min_value, max_value)
        plt.close(fig)
    
    if output_format == 'MP4':
        return Video(file_name)
    elif output_format == 'GIF':
        with open(file_name,'rb') as f:
            display(Image(data=f.read(), format='gif'))
        return None