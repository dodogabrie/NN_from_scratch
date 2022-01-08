import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cimport topology.topology

from topology.topology cimport neuron_t, layer_t, network_t

cdef void feed_input(network_t, double[:])
cdef void forward_prop(network_t)
cdef void update_weights(network_t)
cdef double compute_error(network_t, double[:])
cdef void back_prop(network_t, double [:])
cdef network_t train(network_t, double[:, :], double [:,:], double[:, :],
                     double [:,:], int, int)
cdef double[:,:] predict_out(network_t, double[:,:])
