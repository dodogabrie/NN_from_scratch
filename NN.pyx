# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np # Make numpy work with cython
from libc.stdlib cimport malloc, free
cimport topology
cimport training
from topology cimport network_t, create_network, create_neuron
from training cimport feed_input, forward_prop, back_prop

cdef class network:
    cdef network_t net

    def __init__(self, int[:] structure, np.ndarray activations, double eta):
        self.net = create_network(structure, activations, eta)

    @property
    def eta(self):
        return self.net.eta

    @property
    def num_layers(self):
        return self.net.num_layers

    def feed_input(self, double[:,:] inputs, int i):
        self.net = feed_input(self.net, inputs, i)

    def forward_prop(self):
        self.net = forward_prop(self.net)
    def back_prop(self, double[:] labels):
        self.net = back_prop(self.net, labels)
