import setuptools
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Distutils
import numpy as np

setuptools.setup(
    name = 'fmsne',
    version = '0.4.0',
    description = 'Fast Multi-Scale Neighbour Embedding',
    url = "https://github.com/cdebodt/Fast_Multi-scale_NE",
    packages = ['fmsne'],
    author = 'Cyril de Bodt',
    author_email = 'cyril.debodt@uclouvain.be',
    license = "MIT",
    platforms = ['any'],
    ext_modules = cythonize([
        Extension('fmsne_implem', [
            'fmsne/fmsne_implem.pyx',
            'fmsne/lbfgs.c'
        ])], annotate=False),
    install_requires = [
        'numpy',
        'numba',
        'Cython',
        'matplotlib',
        'sklearn',
        'scipy'],
    setup_requires = [
        'Cython',
        'numpy'],
    include_dirs = [np.get_include(), 'fmsne']
)
