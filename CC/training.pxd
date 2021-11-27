cimport topology
cimport generate_neurons
from topology cimport neuron_t, network_t

cdef void feed_input(network_t, double[:])
cdef void forward_prop(network_t)
cdef void update_weights(network_t)
cdef double compute_error(network_t, double[:])
cdef void back_prop(network_t, double [:])
cdef network_t train(network_t, double[:, :], double [:,:], int, int, double, double, double[:, :],
                     double [:,:], int, int)
cdef double[:,:] predict_out(network_t, double[:,:])
#cdef void add_neuron(network_t*)
