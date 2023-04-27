
import numpy as np
from scipy.sparse import spdiags

from .conservative_dissipative_system import ConservativeDissipativeSystem

__all__ = ['KdVSystem',  'initial_condition_kdv']


class KdVSystem(ConservativeDissipativeSystem):
    """
    Description to be added

    """

    def __init__(self, x=np.linspace(0,20.-.2,100), eta=6., gamma=1., nu=0.,
                 force=None, force_jac=None, init_sampler=None,
                 **kwargs):
        M = x.size
        dx = x[-1]/(M-1)
        e = np.ones(M)
        Dp = 1/dx*spdiags([e,-e,e], np.array([-M+1,0,1]), M, M).toarray() # Forward difference matrix
        D1 = .5/dx*spdiags([e,-e,e,-e], np.array([-M+1,-1,1,M-1]), M, M).toarray() # Central difference matrix
        D2 = 1/dx**2*spdiags([e,e,-2*e,e,e], np.array([-M+1,-1,0,1,M-1]), M, M).toarray() # 2nd order central difference matrix
        skewsymmetric_matrix = D1
        self.skewsymmetric_matrix_flat = .5/dx*np.array([[[-1,0,1]]])
        self.x = x
        self.M = M
        self.eta = eta
        self.gamma = gamma
        self.nu = nu
        self.force = force
        self.force_jac = force_jac
        self.D2 = D2
        self.sample_trajectory = self.sample_trajectory_midpoint
        
        def ham(u):
            return np.sum(-1/6*eta*u**3 + (.5*gamma**2*(np.matmul(Dp,u.T))**2).T, axis=-1)

        def dissintegral(u):
            return np.sum(.5*nu*(np.matmul(Dp,u.T)**2).T, axis=-1)
    
        def ham_grad(u):
            return -.5*eta*u**2 - (gamma**2 * u @ D2)
    
        def dissintegral_grad(u):
            return -nu*u @ D2

        def ham_hessian(u):
            return -eta*np.diag(u) - gamma**2*D2
            
        def dissintegral_hessian(u):
            return -nu*D2
        
        if init_sampler is None:
            init_sampler=initial_condition_kdv(x, eta)

        super().__init__(nstates=M, skewsymmetric_matrix=skewsymmetric_matrix,
                            hamiltonian=ham, dissintegral=dissintegral,
                            grad_hamiltonian=ham_grad,
                            grad_dissintegral=dissintegral_grad,
                            hess_hamiltonian=ham_hessian,
                            hess_dissintegral=dissintegral_hessian,
                            external_forces=force, jac_external_forces=force_jac,
                            init_sampler=init_sampler,
                            **kwargs)


def initial_condition_kdv(x=np.linspace(0,20.-.2,100), eta=6.):
    """
    Decription to be added

    """
    M = x.size
    P = (x[-1]-x[0])*M/(M-1)
    sech = lambda a: 1/np.cosh(a)
    def sampler(rng):
        k1, k2 = rng.uniform(0.5, 2.0, 2)
        d1, d2 = rng.uniform(0., 1., 1), rng.uniform(0., 1., 1)
        u0 = 0
        u0 += (-6./-eta)*2 * k1**2 * sech(k1 * ((x+P/2-P*d1) % P - P/2))**2
        u0 += (-6./-eta)*2 * k2**2 * sech(k2 * ((x+P/2-P*d2) % P - P/2))**2
        u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
        return u0

    return sampler