import numpy as np
cimport numpy as np
ctypedef double (*f_type)(double)

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
    double eta
    layer_t * lay


cdef neuron_t create_neuron(int)
cdef layer_t create_layer(int, f_type, f_type)
cdef network_t create_network(int[:], np.ndarray, double)
