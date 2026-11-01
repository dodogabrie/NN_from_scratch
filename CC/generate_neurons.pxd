cimport topology
from topology cimport neuron_t, network_t


cdef neuron_t generate_candidate(network_t)
cdef network_t add_neuron(network_t, neuron_t)
cdef network_t find_best_candidate(network_t, double[:,:], double[:,:], int, int, double)
