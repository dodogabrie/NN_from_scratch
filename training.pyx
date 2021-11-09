# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport exp
from topology cimport neuron_t, layer_t, network_t

ctypedef np.double_t DOUBLE_t

cdef network_t feed_input(network_t network, double[:,:] inputs, int i):
    cdef int j
    for j in range(network.lay[0].num_neu):
        network.lay[0].neu[j].actv = inputs[i][j]
    return network

cdef network_t forward_prop(network_t network):
    cdef int i, j, k
    cdef double xw
    cdef layer_t lay_here, lay_prev
    for i in range(1, network.num_layers):
        lay_here = network.lay[i]
        lay_prev = network.lay[i-1]
        for j in range(lay_here.num_neu):
            for k in range(lay_prev.num_neu):
                xw = lay_prev.neu[k].out_weights[j] * lay_prev.neu[k].actv
                lay_here.neu[j].net += xw
            lay_here.neu[j].actv = lay_here.activation_f(lay_here.neu[j].net)
        network.lay[i] = lay_here
    return network

cdef network_t back_prop(network_t network, double [:] labels):
    cdef int i, j, k, rev
    cdef int num_layers = network.num_layers
    cdef layer_t lay = network.lay[num_layers-1]
    cdef layer_t lay_back = network.lay[num_layers-2]
    ### Last Layer ###
    for j in range(lay.num_neu):
        lay.neu[j].dnet = (lay.neu[j].actv - labels[j]) * lay.derivative_f(lay.neu[j].net)
        for k in range(lay_back.num_neu):
            lay_back.neu[k].dw[j] += lay.neu[j].dnet * lay_back.neu[k].actv
            lay_back.neu[k].dactv = lay_back.neu[k].out_weights[j] * lay.neu[j].dnet
        lay.neu[j].dbias += lay.neu[j].dnet
    network.lay[num_layers-1] = lay
    network.lay[num_layers-2] = lay_back
    ##################

    ### Inner Layers ###
    for i in range(network.num_layers-2):
        rev = num_layers - 2 - i
        lay = network.lay[rev]
        lay_back = network.lay[rev-1]
        for j in range(lay.num_neu):
            lay.neu[j].dnet = lay.neu[j].dactv * lay.derivative_f(lay.neu[j].net)
            for k in range(lay_back.num_neu):
                lay_back.neu[k].dw[j] += lay_back.neu[k].out_weights[j] * lay.neu[j].dnet
                if rev > 1:
                    lay_back.neu[k].dactv = lay_back.neu[k].out_weights[j] * lay.neu[j].dnet
    ####################
    return network

cdef network_t update_weights(network_t network, double eta):
    cdef int i, j, k
    cdef layer_t layer
    for i in range(network.num_layers-1):
        lay = network.lay[i]
        for j in range(lay.num_neu):
            for k in range(network.lay[i+1].num_neu):
                lay.neu[j].out_weights[k] += eta * lay.neu[j].dw[k]
                lay.neu[j].dw[k] = 0.
            lay.neu[j].bias += eta * lay.neu[j].dbias
            lay.neu[j].dbias = 0.
        network.lay[i] = lay
    return network

cdef double compute_error(network_t network, double[:] labels):
    cdef double error_single, sum_error = 0, quadratic_error
    cdef int i, num_layers = network.num_layers-1
    for i in range(network.lay[num_layers-1].num_neu):
        error_single = labels[i] - network.lay[num_layers-1].neu[i].actv
        quadratic_error = error_single*error_single
        sum_error += quadratic_error
    return sum_error

