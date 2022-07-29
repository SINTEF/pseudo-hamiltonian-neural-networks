import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='porthamiltonians',
    version='0.1.0',
    author='Sølve Eidnes',
    author_email='solve.eidnes@sintef.no',
    description=('A package for simulating and learning port-Hamiltonian systems.'
                 ' For further details, see https://arxiv.org/pdf/2206.02660.pdf'),
    keywords='port Hamiltonian neural networks',
    url="https://gitlab.sintef.no/hybrid-machine-learning/port-hamiltonian-neural-networks",
    packages=setuptools.find_packages(),
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 4 - Beta'
    ],
    license='MIT',
    install_requires=[
        'networkx',
        'numpy',
        'torch',
        'torchaudio',
        'torchvision',
        'scipy'
    ],
)
