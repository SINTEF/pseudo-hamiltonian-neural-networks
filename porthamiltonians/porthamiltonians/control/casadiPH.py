from casadi import *


class CasadiPortHamiltonianSystem:
    """
    Casadi implementation of the port-Hamiltonian framework, modeling a
    system of the form::

        dx/dt = (S - R)*grad(H) + F(x, t)

    where S is the skew-symmetric interconnection matrix,
    R is a diagonal positive semi-definite damping/dissipation-matrix,
    H is the Hamiltonian of the system, F is the external interaction
    and x is the system state. The system dimension is denoted by nstates.

    """
    def __init__(self, S, dH, u, R=None, F=None):
        self.S = S  # S can possibly be dependent on x?
        self.nstates = S.shape[0]
        self.dH = dH
        if R is None:
            self.R = np.zeros((self.nstates, self.nstates))
        elif callable(R):
            self.R = R
        elif len(R.shape) == 1:
            self.R = np.diag(R)
        else:
            self.R = R
        assert self.R.shape == S.shape, ('R must be of size (nstates, nstates),'
                                         f' same as S ({S.shape}), '
                                         f'but is of size {R.shape}.')
        self.F = F
        self.u = u

    def create_forward(self):
        return (self.S - self.R) @ self.dH + self.F + self.u
