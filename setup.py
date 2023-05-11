import setuptools
# from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Distutils
import numpy as np

setuptools.setup(
    name = 'fmsne',
    description = 'Fast Multi-Scale Neighbour Embedding',
    version = '0.3.1',
    packages = ['fmsne'],
    ext_modules = cythonize([
        Extension('fmsne_implem', [
            'fmsne/fmsne_implem.pyx',
            'fmsne/lbfgs.c'
        ])], annotate=False),
    include_dirs = [np.get_include(), 'fmsne']
)
