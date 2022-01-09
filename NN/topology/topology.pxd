import numpy as np
cimport numpy as np
cimport activations.activations
from activations.activations cimport sigmoid, sigmoid_der, lin, lin_der
from libc.stdlib cimport malloc, free
from libc.math cimport exp
cimport topology.topology
from topology.topology cimport neuron_t, layer_t, network_t

ctypedef double (*f_type)(double)
ctypedef np.double_t DOUBLE_t


cdef struct neuron_t:
    double actv
    double * out_weights
    double bias
    double net
    double dactv
    double * dw
    double dbias
    float dnet

cdef struct layer_t:
    int num_neu
    f_type activation_f
    f_type derivative_f
    neuron_t * neu

cdef struct network_t:
    int num_layers
    double eta # learning rate
    double l   # lambda (tickonov)
    double alpha # momentum
    double * train_errors
    double * val_errors
    layer_t * lay


cdef neuron_t create_neuron(int, double)
cdef layer_t create_layer(int, f_type, f_type)
cdef network_t create_network(int[:], np.ndarray, double, double, double, double)
