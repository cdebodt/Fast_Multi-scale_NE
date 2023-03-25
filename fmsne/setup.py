from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    version = "0.0.2",
    name = "fmsne",
    ext_modules = cythonize([Extension("fmsne.fmsne_implem", ["fmsne_implem.pyx", "lbfgs.c"])], annotate=False),
    include_dirs=[np.get_include(), '.']
)
