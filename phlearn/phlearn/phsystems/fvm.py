import numpy as np


# From https://stackoverflow.com/a/39662359
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Burgers:
    def __call__(self, u):
        return 0.5 * u**2

    def du(self, u):
        return u


class BL:
    def __call__(self, u):
        return u**2 / (u**2 + (1 - u) ** 2)

    def du(self, u):
        return (2 * u * (u * u + (1 - u) * (1 - u)) - u * u * (2 * u - 2 * (1 - u))) / (
            (u * u + (1 - u) * (1 - u)) * (u * u + (1 - u) * (1 - u))
        )


def godunov(left, right, f):
    # ONLY WORKS FOR FUNCTIONS WITH MINIMUM at 0.0 (at least for values of the flux function)
    f_left = f(max(left, 0.0))
    f_right = f(min(right, 0.0))

    F = max(f_left, f_right)

    return F


class Dummybar:
    def __init__(self, *args, **kwargs) -> None:
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def refresh(self):
        pass

    def set_postfix(self, *args, **kwargs):
        pass


def solve(
    u0,
    T,
    N,
    M,
    f,
    x_end=1,
    source=lambda x, t: np.zeros_like(x),
    viscosity=0.0,
    progress_bar=Dummybar,
):
    """
    Least efficient way ever to implement FVM in Python
    """

    x, dx = np.linspace(0, x_end, N, retstep=True)
    # Allocate two more for boundary conditions
    u = np.zeros(x.shape[0] + 2)
    if isinstance(u0, np.ndarray):
        u[1:-1] = u0
    elif callable(u0):
        u[1:-1] = u0(x)
    dt = dx / max(2 * abs(f.du(ui)) for ui in u)
    t = 0.0

    u_new = np.zeros_like(u)
    all_u = [u[1:-1].copy()]
    du_dts = []
    t_saves, dt_save = np.linspace(0, T, M, retstep=True)

    next_save_index = 1
    with progress_bar(total=T) as pbar:
        while t < T:
            pbar.n = np.round(t * 1000) / 1000.0
            pbar.refresh()

            du_dt = np.zeros(N)
            central_diff = np.zeros_like(x)
            # This loop should probably be vectorized:
            for i in range(1, u.shape[0] - 1):
                u_new[i] = u[i] - dt / dx * (
                    godunov(u[i], u[i + 1], f) - godunov(u[i - 1], u[i], f)
                )
                du_dt[i - 1] = (
                    -1 / dx * (godunov(u[i], u[i + 1], f) - godunov(u[i - 1], u[i], f))
                )
                central_diff = (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2
            u[1:-1] = u_new[1:-1] + dt * source(x, t) + dt * viscosity * central_diff
            u[0] = u[-2]
            u[-1] = u[1]
            if t == 0:
                du_dts.append(du_dt)

            dt_conservation = dx / max(2 * abs(f.du(ui)) for ui in u)
            if viscosity != 0:  # Want to avoid division by zero
                dt_viscos = dx**2 / viscosity
                pbar.set_postfix(dt=dt, dt_cons=dt_conservation, dt_viscos=dt_viscos)
                dt = min(dt_conservation, dt_viscos)
            else:
                dt = dt_conservation

            if t + dt > t_saves[next_save_index]:
                dt = t_saves[next_save_index] - t

            if t >= t_saves[next_save_index]:
                du_dts.append(du_dt)
                all_u.append(u[1:-1].copy())
                next_save_index += 1
            t += dt
    du_dts.append(du_dt)
    all_u.append(u[1:-1].copy())

    return x, u[1:-1], np.array(all_u), np.array(du_dts)
