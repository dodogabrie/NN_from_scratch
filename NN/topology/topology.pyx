# distutils: language=c++
import numpy as np
cimport numpy as np

cdef neuron_t create_neuron(int num_out_weights, double w_init):
    cdef neuron_t neu # define the neuron
    cdef np.ndarray[DOUBLE_t, ndim=1, mode='c'] rnd_num
    cdef int i
    # Define weights and bias ###########
    rnd_num = np.random.uniform(-w_init,w_init, size = num_out_weights + 1) # array for rnd weights
    neu.out_weights = <double*>malloc(num_out_weights * sizeof(double))
    for i in range(num_out_weights):
        neu.out_weights[i] = rnd_num[i]
    neu.dw = <double*>malloc(num_out_weights * sizeof(double))
    for i in range(num_out_weights):
        neu.dw[i] = 0.
    neu.bias = rnd_num[num_out_weights]
    neu.dbias = 0.
    #####################################
    neu.actv = 0.
    neu.dactv = 0.
    neu.net = 0.
    neu.dnet = 0.
    return neu

cdef layer_t create_layer(int num_of_neurons, f_type activation_f, f_type derivative_f):
    cdef layer_t lay
    lay.num_neu = num_of_neurons
    lay.activation_f = activation_f
    lay.derivative_f = derivative_f
    lay.neu = <neuron_t*>malloc(num_of_neurons * sizeof(neuron_t))
    return lay

cdef network_t create_network(int[:] structure, np.ndarray activations, double eta, double w_init, double alpha, double l):
    cdef int num_of_layers = len(structure)
    cdef int i, j
    cdef network_t net
    net.num_layers = num_of_layers
    net.eta = eta
    net.l = l
    net.alpha = alpha
    # allocate memory
    net.lay = <layer_t*>malloc(num_of_layers * sizeof(layer_t))
    # middle layers/neurons
    for i in range(num_of_layers-1):
        if activations[i] == 'sigmoid':
            net.lay[i] = create_layer(structure[i], sigmoid, sigmoid_der)
        elif activations[i] == 'linear':
            net.lay[i] = create_layer(structure[i], lin, lin_der)
        else: print('Activation func not valid')
        for j in range(structure[i]):
            net.lay[i].neu[j] = create_neuron(structure[i+1], w_init)

    # Outer layers/neuron
    if activations[num_of_layers-1] == 'sigmoid':
        net.lay[num_of_layers-1] = create_layer(structure[num_of_layers-1], sigmoid, sigmoid_der)
    elif activations[num_of_layers-1] == 'linear':
        net.lay[num_of_layers-1] = create_layer(structure[num_of_layers-1], lin, lin_der)
    else: print('Activation func not valid')
    for i in range(structure[num_of_layers-1]):
        net.lay[num_of_layers-1].neu[i] = create_neuron(1, w_init)
    return net

#cdef network_t add_layer(network_t net, f_type activation_f, f_type derivative_f):
#    return net
