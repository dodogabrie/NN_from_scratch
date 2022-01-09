# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np

# Linear activation function
cdef double lin(double)
cdef double lin_der(double)

# Sigmoid activation function
cdef double sigmoid(double)
cdef double sigmoid_der(double)
