cimport topology
from topology cimport neuron_t, layer_t, network_t

cdef network_t feed_input(network_t, double[:,:], int)
cdef network_t forward_prop(network_t)
cdef network_t update_weights(network_t, double)
cdef double compute_error(network_t, double[:])
cdef network_t back_prop(network_t, double [:])
