# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport rand, RAND_MAX
from libc.stdlib cimport malloc, free, realloc
from libc.stdio cimport printf
from libc.math cimport exp, abs
from topology cimport neuron_t, network_t, create_neuron
from generate_neurons cimport find_best_candidate

ctypedef np.double_t DOUBLE_t


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef network_t train(network_t net, double[:, :] inputs, double [:,:] labels, int patience, int max_hidden, double err_solved, double max_plateau,
                     double[:,:] validation, double[:,:] labels_validation, int epoch, int val_data):
    cdef int i, j, k
    cdef int count_patience = 0
    cdef int n_data = len(inputs)
    cdef int n_val_data = len(validation)
    cdef double inv_n_data = 1/<double>n_data
    cdef double inv_n_val = 1/<double>n_val_data
    cdef double error # Temporary error for each train sample
    # Initialize array of errors
    net.train_errors = <double*>malloc(epoch*sizeof(double))
    net.val_errors = <double*>malloc(epoch*sizeof(double))
    for i in range(epoch): # for each epoch
#        printf("Epoch %d\r", i)
        # Train the model
        for j in range(n_data): # for each data
            feed_input(net, inputs[j])
            forward_prop(net)
            error += compute_error(net, labels[j])
            back_prop(net, labels[j])
        # Batch mode: update after I see all the dataset
        update_weights(net)
        net.train_errors[i] = error*inv_n_data
        error = 0

        # Compute error on validation dataset
        if val_data:
            for k in range(n_val_data):
                feed_input(net, validation[k])
                forward_prop(net)
                error += compute_error(net, labels_validation[k])
            net.val_errors[i] = error*inv_n_val
            error = 0

        # Add a neuron if plateau reached
#        if i > 0:
#            d_err = net.train_errors[i] - net.train_errors[i-1]
#            if (abs(d_err) < max_plateau) and (net.num_hidden < max_hidden):
#                count_patience +=1
#        if count_patience > patience:
#            print('PATIENCE LOSTTT')

    printf("Done!       \r")
    printf("\n")
    net = find_best_candidate(net, inputs, labels, 10, 1)
    return net

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void feed_input(network_t net, double[:] inputs):
    cdef int i
    for i in range(net.num_inputs):  net.inputs[i] = inputs[i]

cdef void forward_prop(network_t net):
    cdef int i, j, k, cnt_out = 0
    cdef int n = net.num_inputs, nh = net.num_hidden
    cdef double xw
    cdef neuron_t * neu_i
    for i in range(net.tot_neurons):
        neu_i = &net.neu[i]
        neu_i.net = neu_i.bias
        for j in range(neu_i.num_w_in): neu_i.net += neu_i.w_in[j] * neu_i.inputs[j]
        neu_i.actv = neu_i.actv_f(neu_i.net)
        if i > net.num_hidden: cnt_out += 1
        for j in range(i+1, net.tot_neurons):
            net.neu[j-cnt_out].inputs[net.num_inputs + i] = neu_i.actv


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void back_prop(network_t net, double [:] labels):
    cdef int i, j
    cdef n_hidd = net.num_hidden, n_out = net.num_outputs
    cdef neuron_t * neu_i
    for i in range(n_hidd, n_hidd + n_out):
        neu_i = &net.neu[i]
        neu_i.dnet = (labels[i]-neu_i.actv) * neu_i.actv_df(neu_i.net)
        for j in range(neu_i.num_w_in): neu_i.dw_in[j] += neu_i.dnet * neu_i.inputs[j]
        neu_i.dbias += neu_i.dnet

cdef void update_weights(network_t net):
    cdef int i, j, k
    for i in range(net.tot_neurons): # for each neuron in the layer
        for j in range(net.neu[i].num_w_in): # for neu in the next lay
            net.neu[i].w_in[j] += net.eta * net.neu[i].dw_in[j]
            net.neu[i].dw_in[j] = 0.
        net.neu[i].bias += net.eta * net.neu[i].dbias
        net.neu[i].dbias = 0

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double[:,:] predict_out(network_t net, double[:,:] data):
    cdef int i, j
    cdef int n_data = len(data)
    cdef int n_out_features = net.num_outputs
    # Initialize array of predictions
    cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] predictions = np.empty((n_data, n_out_features))
    for i in range(n_data):
        feed_input(net, data[i]) # Send in the first layer the data
        forward_prop(net)        # Propagate it in the net
        for j in range(n_out_features): # Save the result in predictions
            predictions[i, j] = net.neu[net.num_hidden + j].actv
    return predictions


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double compute_error(network_t net, double[:] labels):
    cdef double error_single, sum_error = 0, quadratic_error
    cdef int i
    for i in range(net.num_outputs):
        error_single = labels[i] - net.neu[net.num_hidden + i].actv
        quadratic_error = error_single*error_single
        sum_error += quadratic_error
    return sum_error

