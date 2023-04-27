
import numpy as np
from scipy.sparse import spdiags

from .conservative_dissipative_system import ConservativeDissipativeSystem

__all__ = ['AllenCahnSystem', 'init_ac', 'initial_condition_ac']

class AllenCahnSystem(ConservativeDissipativeSystem):
    """
    Description to be added

    """

    def __init__(self, x=np.linspace(0, 6.-6/100, 100), nu=.001, force=None,
                 force_jac=None, **kwargs):
        M = x.size
        dx = x[-1]/(M-1)
        e = np.ones(M)
        Dp = 1/dx*spdiags([e, -e, e], np.array([-M+1, 0, 1]),
                          M, M).toarray()  # Forward difference matrix
        # Central difference matrix
        D1 = .5/dx*spdiags([e, -e, e, -e],
                           np.array([-M+1, -1, 1, M-1]), M, M).toarray()
        # 2nd order central difference matrix
        D2 = 1/dx**2*spdiags([e, e, -2*e, e, e],
                             np.array([-M+1, -1, 0, 1, M-1]), M, M).toarray()
        I = np.eye(M)
        skewsymmetric_matrix = D1
        self.skewsymmetric_matrix_flat = .5/dx*np.array([[[-1, 0, 1]]])
        self.x = x
        self.M = M
        self.force = force
        self.force_jac = force_jac
        self.D2 = D2

        def ham(u):
            return np.sum(np.zeros_like(u), axis=1)

        def dissintegral(u):
            return 1/2*np.sum((nu*np.matmul(Dp, u.T)**2).T - u**2 + 1/2*u**4, axis=1)

        def ham_grad(u):
            return np.zeros_like(u)

        def dissintegral_grad(u):
            return -nu*u @ D2 - u + u**3

        def ham_hessian(u):
            return np.zeros_like(D1)

        def dissintegral_hessian(u):
            return -nu*D2 - I + 3*np.diag(u**2)

        super().__init__(nstates=M, skewsymmetric_matrix=skewsymmetric_matrix,
                         hamiltonian=ham, dissintegral=dissintegral,
                         grad_hamiltonian=ham_grad,
                         grad_dissintegral=dissintegral_grad,
                         hess_hamiltonian=ham_hessian,
                         hess_dissintegral=dissintegral_hessian,
                         external_forces=force, jac_external_forces=force_jac,
                         **kwargs)


def init_ac():
    """
    Initialize a standard Allen-Cahn system

    Returns
    -------
    AllenCahnSystem

    """
    x_end = 6
    x_points = 300
    dx = x_end/x_points
    x = np.linspace(0, x_end-dx, x_points)

    return AllenCahnSystem(x=x, init_sampler=initial_condition_ac(x))


def initial_condition_ac(x=np.linspace(0, 6.-6/100, 100)):
    """
    Add description (see ODE examples)

    """
    M = x.size
    P = (x[-1]-x[0])*M/(M-1)

    def sampler(rng):
        d1, d2 = rng.uniform(0., 1., 2)
        k1, k2 = rng.uniform(0.7, 1., 2)
        u0 = 0
        u0 += k1*np.cos(2*np.pi/P*(x-d1*P))
        u0 += k2*np.cos(2*np.pi/P*(x-d2*P))
        return u0

    return sampler
