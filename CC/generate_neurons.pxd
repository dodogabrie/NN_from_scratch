cimport topology
from topology cimport neuron_t, network_t


cdef neuron_t generate_candidate(network_t)
cdef network_t add_space_for_neuron(network_t)
cdef network_t find_best_candidate(network_t, double[:,:], double[:,:], int, double)
cdef double compute_S(network_t, neuron_t, double[:,:] data, double[:,:] labels)
cdef neuron_t manually_connect_neu(network_t, neuron_t)
cdef double compute_mean_out(network_t, neuron_t, int, double[:,:], double[:,:])
cdef double compute_mean_error(network_t, int, double[:,:], double[:,:])
