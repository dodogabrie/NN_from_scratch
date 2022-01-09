import numpy as np
cimport cython
cimport numpy as np
cimport topology.topology
from topology.topology cimport neuron_t, layer_t, network_t
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport exp, sqrt

ctypedef np.double_t DOUBLE_t


cdef void feed_input(network_t, double[:])
cdef void forward_prop(network_t)
cdef void update_weights(network_t)
cdef double compute_error(network_t, double[:])
cdef void back_prop(network_t, double [:])
cdef network_t train(network_t, double[:, :], double [:,:], double[:, :],
                     double [:,:], int, int)
cdef double[:,:] predict_out(network_t, double[:,:])
