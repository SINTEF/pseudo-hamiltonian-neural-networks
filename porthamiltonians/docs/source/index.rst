
==========================================
Welcome to porthamiltonians' documentation
==========================================

porthamiltonians is a python package for modelling port-Hamiltonian systems with neural networks as decribed in `(Eidnes et al. 2022) <https://arxiv.org/pdf/2206.02660.pdf>`_.

Installation
============

The porthamiltonian package is available via PyPi:

::

   $ pip install porthamiltonians

Alternatively, to get the latest updates not yet available on PyPi, you can clone the repositrory from Github and install directly:

::

   $ git clone https://github.com/SINTEF/port-hamiltonian-neural-networks.git
   $ cd port-hamiltonian-neural-networks
   $ pip install -e porthamiltonians


Example use
===========
Example scripts and notebooks can be found in `the Github repo <https://github.com/SINTEF/port-hamiltonian-neural-networks/tree/main/example_scripts>`_.


porthamiltonians API
====================

.. automodule:: porthamiltonians

.. toctree::
   :maxdepth: 4
   :hidden:

   phsystems
   phnns
   control
   utils
