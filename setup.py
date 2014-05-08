# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:03:37 2014

@author: Ray
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("RadialAverage_C.pyx", include_dirs = [np.get_include()]),
)