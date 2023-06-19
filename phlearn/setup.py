import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(
    name='phlearn',
    version='1.1.2',
    author='Sølve Eidnes',
    author_email='solve.eidnes@sintef.no',
    description=('A package for simulating and learning pseudo-Hamiltonian systems.'
                 ' For further details, see https://arxiv.org/pdf/2206.02660.pdf and https://arxiv.org/abs/2304.14374'),
    keywords='pseudo-Hamiltonian neural networks',
    url="https://github.com/SINTEF/pseudo-hamiltonian-neural-networks",
    packages=setuptools.find_packages(),
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 5 - Production/Stable'
    ],
    license='MIT',
    install_requires=[
        'networkx>=2.7.1',
        'numpy>=1.22.3',
        'torchvision',
        'scipy>=1.8.0',
        'torch',
        'torchaudio',
        'matplotlib',
        'imageio',
        'tqdm',
        'autograd',
        'IPython',
    ],
    extras_require={'control': ['casadi', 'do_mpc']},
)
