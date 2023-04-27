
import numpy as np
import numpy.linalg as la
from scipy.sparse import spdiags

from .conservative_dissipative_system import ConservativeDissipativeSystem

__all__ = ['CahnHilliardSystem', 'init_ch',
           'initial_condition_ch']


class CahnHilliardSystem(ConservativeDissipativeSystem):
    """
    Description to be added

    """

    def __init__(self, x=np.linspace(0,1.-1/100,100), nu=-1., alpha= 1.,
                 mu=-.001, force=None, force_jac=None, **kwargs):
        M = x.size
        dx = x[-1]/(M-1)
        e = np.ones(M)
        Dp = 1/dx*spdiags([e,-e,e], np.array([-M+1,0,1]), M, M).toarray() # Forward difference matrix
        D1 = .5/dx*spdiags([e,-e,e,-e], np.array([-M+1,-1,1,M-1]), M, M).toarray() # Central difference matrix
        D2 = 1/dx**2*spdiags([e,e,-2*e,e,e], np.array([-M+1,-1,0,1,M-1]), M, M).toarray() # 2nd order central difference matrix
        I = np.eye(M)
        skewsymmetric_matrix = D1
        dissipation_matrix = -D2
        self.skewsymmetric_matrix_flat = .5/dx*np.array([[[-1,0,1]]])
        self.dissipation_matrix_flat = -1/dx**2*np.array([[[1,-2,1]]])
        self.x = x
        self.M = M
        self.force = force
        self.force_jac = force_jac
        self.D2 = D2
        self.sample_trajectory = self.sample_trajectory_midpoint

        def dissintegral(u):
            return 1/2*np.sum(nu*u**2 + 1/2*alpha*u**4 - mu*(np.matmul(Dp,u.T)**2).T, axis=1)

        def dissintegral_grad(u):
            return nu*u + alpha*u**3 + mu*u@D2

        def ham_hessian(u):
            return np.zeros_like(D1)
        
        def dissintegral_hessian(u):
            return nu*I + 3*alpha*np.diag(u**2) + mu*D2

        super().__init__(nstates=M, skewsymmetric_matrix=skewsymmetric_matrix,
                            dissipation_matrix=dissipation_matrix,
                            dissintegral=dissintegral,
                            grad_dissintegral=dissintegral_grad,
                            hess_hamiltonian=ham_hessian,
                            hess_dissintegral=dissintegral_hessian,
                            external_forces=force, jac_external_forces=force_jac,
                            **kwargs)


def init_ch():
    """
    Initialize a standard Cahn-Hilliard system

    Returns
    -------
    CahnHilliardSystem

    """
    x_end = 1
    x_points = 100
    dx = x_end/x_points
    x = np.linspace(0,x_end-dx,x_points)

    return CahnHilliardSystem(x=x, init_sampler=initial_condition_ch(x))


def initial_condition_ch(x=np.linspace(0,1.-1/100,100)):
    """
    Description to be added

    """
    M = x.size
    P = (x[-1]-x[0])*M/(M-1)
    def sampler(rng):
        a1, a2 = rng.uniform(0., .05, 2)
        a3, a4 = rng.uniform(0., .2, 2)
        k1, k2, k3, k4 = rng.integers(1, 6, 4)
        u0 = 0
        u0 += a1*np.cos(2*k1*np.pi/P*x)
        u0 += a2*np.cos(2*k2*np.pi/P*x)
        u0 += a3*np.sin(2*k3*np.pi/P*x)
        u0 += a4*np.sin(2*k4*np.pi/P*x)
        return u0

    return sampler