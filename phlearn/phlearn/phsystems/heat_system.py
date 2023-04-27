
import numpy as np
from scipy.sparse import spdiags

from .conservative_dissipative_system import ConservativeDissipativeSystem # Should be changed to preservation-dissipation system

__all__ = ['HeatEquationSystem', 'init_heat',
           'initial_condition_heat']


class HeatEquationSystem(ConservativeDissipativeSystem):
    """
    Decription to be added

    """

    def __init__(self, x=np.linspace(0,6.-6/100,100), nu=1., force=None,
                 force_jac=None, **kwargs):
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
        self.force = force
        self.force_jac = force_jac
        self.D2 = D2
        
        def ham(u):
            return np.sum(np.zeros_like(u), axis=1) # Fix

        def dissintegral(u):
            return np.sum(.5*nu*(np.matmul(Dp,u.T)**2).T, axis=1)
    
        def ham_grad(u):
            return np.zeros_like(u)

        def dissintegral_grad(u):
            return -nu*u @ D2

        def ham_hessian(u):
            return np.zeros_like(D1)
        
        def dissintegral_hessian(u):
            return -nu*D2

        super().__init__(nstates=M, skewsymmetric_matrix=skewsymmetric_matrix,
                            hamiltonian=ham, dissintegral=dissintegral,
                            grad_hamiltonian=ham_grad,
                            grad_dissintegral=dissintegral_grad,
                            hess_hamiltonian=ham_hessian,
                            hess_dissintegral=dissintegral_hessian,
                            external_forces=force, jac_external_forces=force_jac,
                            **kwargs)

def init_heat():
    """
    Initialize a standard heat equation system

    Returns
    -------
    HeatEquationSystem

    """
    x_end = 6
    x_points = 300
    dx = x_end/x_points
    x = np.linspace(0,x_end-dx,x_points)

    return HeatEquationSystem(x=x, init_sampler=initial_condition_heat(x))


def initial_condition_heat(x=np.linspace(0,6.-6/100,100)):
    """
    Decription to be added

    """
    M = x.size
    P = (x[-1]-x[0])*M/(M-1)
    def sampler(rng):
        d1, d2 = rng.uniform(0.3, 3, 2)
        c1, c2 = rng.uniform(0.5, 1.5, 2)
        k1 = rng.uniform(0.5, 3.0, 1)
        k2 = rng.uniform(10., 20., 1)
        n1 = rng.uniform(20., 40., 1)
        n2 = rng.uniform(.05, .15, 1)
        u0 = 0
        u0 += rng.uniform(-5., 5., 1) - c1*np.tanh(n1*(x-d1)) + c1*np.tanh(n1*(x-P+d1))
        u0 += - c2*np.tanh(n1*(x-d2)) + c2*np.tanh(n1*(x-P+d2))
        u0 += n2*np.sin(k1*np.pi*x)**2*np.sin(k2*np.pi*x)
        return u0

    return sampler