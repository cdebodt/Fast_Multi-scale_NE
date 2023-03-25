from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    version = "0.0.1",
    name = "cython_implem",
    ext_modules = cythonize([Extension("cython_implem", ["cython_implem.pyx", "lbfgs.c"])], annotate=False),
    include_dirs=[np.get_include(), '.']
)
