
==========================================
Welcome to phlearn' documentation
==========================================

phlearn is a python package for modelling pseudo-Hamiltonian systems with neural networks as decribed in `(Eidnes et al. 2022) <https://arxiv.org/pdf/2206.02660.pdf>`_.

Installation
============

The phlearn package is available via PyPi:

::

   $ pip install phlearn

Alternatively, to get the latest updates not yet available on PyPi, you can clone the repositrory from Github and install directly:

::

   $ git clone https://github.com/SINTEF/pseudo-Hamiltonian-neural-networks.git
   $ cd pseudo-Hamiltonian-neural-networks
   $ pip install -e phlearn


Example use
===========
Example scripts and notebooks can be found in `the Github repo <https://github.com/SINTEF/pseudo-Hamiltonian-neural-networks/tree/main/example_scripts>`_.


phlearn API
====================

.. automodule:: phlearn

.. toctree::
   :maxdepth: 4
   :hidden:

   phsystems
   phnns
   control
   utils
