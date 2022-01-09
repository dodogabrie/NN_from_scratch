# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport exp

# Linear activation function
cdef double lin(double x): return x
cdef double lin_der(double x): return 1.

# Sigmoid activation function
cdef double sigmoid(double x): return 1/(1 + exp(-x))
cdef double sigmoid_der(double x): return exp(-x)/((1 + exp(-x))*(1 + exp(-x)))
