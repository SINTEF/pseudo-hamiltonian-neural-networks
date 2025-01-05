from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags_array

import phlearn.phnns as phnn
from phlearn.phsystems.pde import PseudoHamiltonianPDESystem


def setup_KdV_system(
    x: np.ndarray,
    eta: float = 6.0,
    gamma: float = 1.0,
    nu: float = 0.0,
    n_solitons: int = 2,
) -> PseudoHamiltonianPDESystem:
    M = x.size
    dx = (x[-1] - x[0]) / (M - 1)

    # Forward difference matrix
    Dp = diags_array([1, -1, 1], offsets=[-M + 1, 0, 1], shape=(M, M))
    Dp = 1 / dx * Dp
    # Central difference matrix
    D1 = diags_array([1, -1, 1, -1], offsets=[-M + 1, -1, 1, M - 1], shape=(M, M))
    D1 = 0.5 / dx * D1
    # 2nd order central difference matrix
    D2 = diags_array([1, 1, -2, 1, 1], offsets=[-M + 1, -1, 0, 1, M - 1], shape=(M, M))
    D2 = 1 / dx**2 * D2

    def hamiltonian(u: np.ndarray) -> np.ndarray:
        integrand = -1 / 6 * eta * u**3 + 0.5 * gamma**2 * ((Dp @ u.T).T ** 2)
        return np.sum(integrand, axis=-1)

    def hamiltonian_grad(u: np.ndarray) -> np.ndarray:
        return -0.5 * eta * u**2 - (gamma**2 * u @ D2)

    def initial_condition() -> Callable[[np.random.Generator], np.ndarray]:
        P = (x[-1] - x[0]) * M / (M - 1)
        sech = lambda a: 1 / np.cosh(a)

        def sampler_train(rng: np.random.Generator) -> np.ndarray:
            u0 = -np.cos(np.pi * x)
            u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
            return u0

        def sampler_solitons(rng: np.random.Generator) -> np.ndarray:
            ks = rng.uniform(0.5, 2.0, n_solitons)
            ds = rng.uniform(0.0, 1.0, n_solitons)
            u0 = np.zeros_like(x)

            for k, d in zip(ks, ds):
                center = (x + P / 2 - P * d) % P - P / 2
                coeff = 6.0 / eta * 2 * k**2
                u0 += coeff * sech(np.abs(k * center)) ** 2

            u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
            return u0

        if n_solitons == 0:
            return sampler_train

        return sampler_solitons

    KdV_system = PseudoHamiltonianPDESystem(
        nstates=M,
        skewsymmetric_matrix=D1,
        hamiltonian=hamiltonian,
        grad_hamiltonian=hamiltonian_grad,
        init_sampler=initial_condition(),
    )

    return KdV_system


def generate_KdV_data(
    x: np.ndarray,
    t_axis: np.ndarray,
    eta: float = 6.0,
    gamma: float = 1.0,
    n_solitons: int = 2,
    n_trajectories: int = 5,
):
    system = setup_KdV_system(x, eta, gamma, n_solitons=n_solitons)
    traindata = phnn.generate_dataset(system, n_trajectories, t_axis, xspatial=x)

    return system, traindata, t_axis


def setup_baseline_nn(spatial_points: int, **kwargs) -> phnn.DynamicSystemNN:
    baseline_nn = phnn.PDEBaselineNN(spatial_points, **kwargs)
    basemodel = phnn.DynamicSystemNN(baseline_nn, baseline_nn)
    return basemodel


def plot_solutions(
    u_exact: np.ndarray,
    u_model: np.ndarray,
    x: np.ndarray,
    t_axis: np.ndarray,
) -> None:
    N = u_exact.shape[0]
    fig = plt.figure(figsize=(7, 4))
    lw = 2
    colors = [
        # (0, 0.4, 1),
        (1, 0.7, 0.3),
        (0.2, 0.7, 0.2),
        (0.8, 0, 0.2),
        (0.5, 0.3, 0.9),
    ]

    timesteps = [int(i * N / 4) for i in range(1, 4)] + [-1]

    plt.plot(x, u_exact[0], "k--", linewidth=lw, label=f"$t = {t_axis[0]:.2f}$")
    for i, idx in enumerate(timesteps):
        plt.plot(
            x,
            u_exact[idx],
            color=colors[i],
            linestyle="--",
            linewidth=lw,
            label=f"$t = {t_axis[idx]:.2f}$, true",
        )
        plt.plot(
            x,
            u_model[idx],
            color=colors[i],
            linestyle="-",
            linewidth=lw,
            label=f"$t = {t_axis[idx]:.2f}$, model",
        )

    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$u$", fontsize=12)
    plt.title("PHNN vs ground truth", fontsize=14)
    plt.legend()
    plt.show()
