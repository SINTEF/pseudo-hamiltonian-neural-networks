
import numbers

import networkx as nx
import numpy as np

from .port_Hamiltonian_system import PortHamiltonianSystem

__all__ = ['TankSystem', 'init_tanksystem',
           'init_tanksystem_leaky']


class TankSystem(PortHamiltonianSystem):
    """
    Implements a port-Hamiltonian version of a coupled tanks system::

          .      | -R_p   B^T |
          x   =  |            | * grad[H(x)] + F(x, t)
                 |  -B     0  |


    where the state x = [phi, mu], phi and mu being proportional to
    pipe flows and tank levels, respectively. The interconnection of the
    tanks is described by a directed graph where each vertex is a tank
    and each edge is a pipe between two tanks. The incidence matrix of
    this graph corresponds to the matrix B.
    The number of tanks is denoted by ntanks, the number of pipes by
    npipes. The number of states is denoted by
    nstates = npipes + ntanks.

    External interaction can be specified for every state, but the most
    physically interpretable setting is to only have external
    interaction for the states corresponding to the tanks, which can be
    seen as volumnetric flows into or out of the tanks.

    Each tank is assumed to have a uniform cross-section w.r.t. it's
    height.

    The interconnection of tanks and pipes can either be specified by a
    directed graph or by an interconnection matrix S.

    parameters
    ----------
        incidence_matrix : (N, N) ndarray, default None
            Incidence matrix of the graph desctibin the tank system.
            Corresponds to the B matrix.  Inferred from system_graph if
            system_graph is provided.

        system_graph : networkx.Graph, default None
            networkx directed graph describing the interconnection of
            the tanks. The graph has ntanks vertices and npipes edges.

        npipes : int, default None
            Number of pipes. Inferred from system_graph if system_graph
            is provided.

        ntanks : int, default None
            Number of tanks. Inferred from system_graph if system_graph
            is provided.
        dissipation_pipes : ndarray, default None
            ndarray of size (npipes,) or (npipes, npipes), describing
            energy loss in the pipes. Corresponds to R_p. Defaults to
            zero if not provided.
        J : number, default 1.
            Scalar or ndarray of size (npipes,) of proportionality
            constants relating the change in volumetric flow through
            each pipe to the pressure drop and potential friction
            through each pipe. If scalar, the same constant is
            used for all pipes.
        A : (ntanks,) ndarray or number, default 1.
         Scalar ndarray of size (ntanks,) of tank cross-section areas.
         If scalar, the same area is  used for all tanks.
        rho : number, default 1.
            Positive scalar. Density of liquid.
        g : number, default 9.81
            Positive scalar. Gravitational acceleration.
        kwargs : any
            Keyword arguments passed to PortHamiltonianSystem
            constructor.

    """

    def __init__(self, incidence_matrix=None, system_graph=None, npipes=None,
                 ntanks=None, dissipation_pipes=None, J=1., A=1., rho=1.,
                 g=9.81, **kwargs):
        self.system_graph = system_graph

        if system_graph is not None:
            self.npipes = system_graph.number_of_edges()
            self.ntanks = system_graph.number_of_nodes()
            B = np.array(nx.linalg.graphmatrix.incidence_matrix(
                system_graph, oriented=True).todense())
        else:
            self.npipes = npipes
            self.ntanks = ntanks
            B = incidence_matrix

        structure_matrix = np.block(
            [[np.zeros((self.npipes, self.npipes)), B.T],
             [-B, np.zeros((self.ntanks, self.ntanks))]])

        nstates = self.npipes + self.ntanks

        if dissipation_pipes is None:
            dissipation_pipes = np.zeros((nstates, nstates))
        elif len(dissipation_pipes.shape) == 1:
            dissipation_pipes = np.diag(dissipation_pipes)

        dissipation = np.block(
            [[dissipation_pipes, np.zeros([self.npipes, self.ntanks])],
             [np.zeros([self.ntanks, self.npipes]),
              np.zeros([self.ntanks, self.ntanks])]])

        if isinstance(J, numbers.Number):
            J = J*np.ones(self.npipes)
        if isinstance(A, numbers.Number):
            A = A*np.ones(self.ntanks)

        self.Hvec = np.concatenate((1/J, rho*g / A))
        super().__init__(nstates, structure_matrix=structure_matrix,
                         dissipation_matrix=dissipation, **kwargs)
        self.dH = self.dH_tanksystem

    def H_tanksystem(self, x, t=None):
        return x**2 @ self.Hvec / 2

    def dH_tanksystem(self, x, t=None):
        return x * self.Hvec

    def pipeflows(self, x):
        return x[..., :self.npipes]

    def tanklevels(self, x):
        return x[..., self.npipes:]


def init_tanksystem(u=None):
    """
    Initialize standard tank system.

    Parameters
    ----------
    u : porthamiltonians.control.PortHamiltonianController, defult None

    Returns
    -------
    TankSystem

    """

    G_s = nx.DiGraph()
    G_s.add_edge(1, 2)
    G_s.add_edge(2, 3)
    G_s.add_edge(3, 4)
    G_s.add_edge(1, 3)
    G_s.add_edge(1, 4)

    npipes = G_s.number_of_edges()
    ntanks = G_s.number_of_nodes()
    nstates = npipes + ntanks
    R = 1.e-2*np.diag(np.array([3., 3., 9., 3., 3.]))
    J = 2.e-2*np.ones(npipes)
    A = np.ones(ntanks)
    ext_filter = np.zeros(nstates)
    ext_filter[-1] = 1

    def F(x, t=None):
        return -1.e1*np.fmin(0.3, np.fmax(x, -0.3))*ext_filter

    return TankSystem(system_graph=G_s, dissipation_pipes=R, J=J, A=A,
                      external_port=F, controller=u)


def init_tanksystem_leaky(nleaks=0):
    """
    Initialize tank system with a leaks.

    Parameters
    ----------
    nleaks : int, default 0
        If 0, no leaks. If 1, there is a leak on the last tank. If 2,
        there is a leak on the last and tank number 2.

    Returns
    -------
    TankSystem

    """

    G_s = nx.DiGraph()
    G_s.add_edge(1, 2)
    G_s.add_edge(2, 3)
    G_s.add_edge(3, 4)
    G_s.add_edge(1, 3)
    G_s.add_edge(1, 4)

    npipes = G_s.number_of_edges()
    ntanks = G_s.number_of_nodes()
    nstates = npipes + ntanks
    R = 1.e-2*np.diag(np.array([3., 3., 9., 3., 3.]))
    J = 2.e-2*np.ones(npipes)
    A = np.ones(ntanks)

    if nleaks == 0:
        def F(x, t=None):
            return np.zeros_like(x)
    else:
        if nleaks == 1:
            ext_filter = np.zeros(nstates)
            ext_filter[-1] = 3
        else:
            ext_filter = np.zeros(nstates)
            ext_filter[-1] = 3
            ext_filter[-4] = 1

        def F(x, t=None):
            return -1.e1*np.minimum(0.3, np.maximum(x, -0.3))*ext_filter

    return TankSystem(system_graph=G_s, dissipation_pipes=R, J=J, A=A,
                      external_port=F, controller=None)
