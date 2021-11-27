import numpy as np
cimport numpy as np
ctypedef double (*f_type)(double)

cdef struct neuron_t:
    f_type actv_f # Activation function of the neuron
    f_type actv_df # Derivative of the activation funtion of the neuron
    double actv # out value of the neuron
    double * w_in # output weights
    double * inputs
    double bias # Bias of the neuron
    double net # net function of the neuron
    double dactv #
    double * dw_in # Variaton of the wout
    double dbias # Variation of the bias
    double dnet # delta
    int num_w_in

cdef struct network_t:
    int num_inputs # Number of inputs units
    int num_hidden # Number of hidden units
    int num_outputs # Number of output units
    int tot_neurons # Total number of neurons
    double * inputs
    double w_init # Init value of weights
    double eta # learning rate
    double l   # lambda (tickonov)
    double alpha # momentum
    double * train_errors # error on train data
    double * val_errors # error on validation data
    neuron_t * neu # input units
    f_type hidden_actv_f
    f_type output_actv_f
    f_type hidden_actv_df
    f_type output_actv_df



cdef neuron_t create_neuron(int, double, f_type, f_type)
cdef network_t create_network(int, int, str, str, double, double)
