import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='phlearn',
    version='1.0.0',
    author='Sølve Eidnes',
    author_email='solve.eidnes@sintef.no',
    description=('A package for simulating and learning pseudo-Hamiltonian systems.'
                 ' For further details, see https://arxiv.org/pdf/2206.02660.pdf'),
    keywords='pseudo-Hamiltonian neural networks',
    url="https://gitlab.sintef.no/hybrid-machine-learning/pseudo-Hamiltonian-neural-networks",
    packages=setuptools.find_packages(),
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 5 - Production/Stable'
    ],
    license='MIT',
    install_requires=[
        'networkx',
        'numpy',
        'torch',
        'torchaudio',
        'torchvision',
        'scipy',
        'casadi',
        'do_mpc'
    ],
)
