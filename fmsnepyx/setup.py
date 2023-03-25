from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
    version = "0.0.3",
    name = "fmsnepyx",
    ext_modules = cythonize([Extension("fmsnepyx", ["fmsne_implem.pyx", "lbfgs.c"])], annotate=False),
    include_dirs=[np.get_include(), '.']
)
