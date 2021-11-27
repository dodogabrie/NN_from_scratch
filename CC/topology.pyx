# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free, realloc
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf
from libc.math cimport exp
from topology cimport neuron_t, network_t

ctypedef np.double_t DOUBLE_t

# Linear activation function
cdef double lin(double x): return x
cdef double lin_der(double x): return 1.

# Sigmoid activation function
cdef double sigmoid(double x): return 1/(1 + exp(-x))
cdef double sigmoid_der(double x): return exp(-x)/((1 + exp(-x))*(1 + exp(-x)))

#cdef neuron_t new_new = create_neuron(net.num_inputs + net.num_hidden,
#                                          net.w_init, net.hidden_activation)

cdef neuron_t create_neuron(int num_inputs_w, double w_init,
                            f_type actv_func, f_type actv_dfunc):
    cdef neuron_t neu # define the neuron
    cdef np.ndarray[DOUBLE_t, ndim=1, mode='c'] rnd_num
    cdef int i
    # Define weights and bias ###########
    rnd_num = np.random.uniform(-w_init, w_init, size = num_inputs_w + 1)
    neu.w_in = <double*>malloc(num_inputs_w * sizeof(double))
    neu.dw_in = <double*>malloc(num_inputs_w * sizeof(double))
    for i in range(num_inputs_w):
        neu.w_in[i] = rnd_num[i]
        neu.dw_in[i] = 0.
    neu.bias = rnd_num[num_inputs_w]
    neu.dbias = 0.
    #####################################
    neu.actv_f = actv_func
    neu.actv_df = actv_dfunc
    neu.inputs = <double*>malloc(num_inputs_w*sizeof(double))
    neu.num_w_in = num_inputs_w
    neu.actv = 0.
    neu.dactv = 0.
    neu.net = 0.
    neu.dnet = 0.
    return neu

cdef network_t create_network(int num_inputs, int num_outputs,
                              str out_activation, str hidden_activation,
                              double eta, double w_init):
    cdef int i, j
    cdef network_t net
    net.eta = eta
    net.w_init = w_init
    # Input "layer"
    net.num_inputs = num_inputs
    # Output "layer"
    net.num_outputs = num_outputs
    net.tot_neurons = num_outputs
    net.neu = <neuron_t*>malloc(num_outputs * sizeof(neuron_t))
    net.num_hidden = 0
    if out_activation == 'linear':
        net.output_actv_f = lin
        net.output_actv_df = lin_der
    elif out_activation == 'sigmoid':
        net.output_actv_f = sigmoid
        net.output_actv_df = sigmoid_der
    else: print('out activation function not found')
    if hidden_activation == 'linear':
        net.hidden_actv_f = lin
        net.hidden_actv_df = lin_der
    elif out_activation == 'sigmoid':
        net.hidden_actv_f = sigmoid
        net.hidden_actv_df = sigmoid_der
    else: print('hidden activation function not found')
    # Create the minimal net
    net.inputs = <double*>malloc(net.num_inputs*sizeof(double))
    for i in range(num_outputs):
        net.neu[i] = create_neuron(net.num_inputs, w_init, net.output_actv_f, net.output_actv_df)
        for j in range(net.num_inputs):
            net.neu[i].inputs = net.inputs
    return net
