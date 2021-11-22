# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport exp
from topology cimport neuron_t, layer_t, network_t

ctypedef np.double_t DOUBLE_t

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void feed_input(network_t network, double[:] inputs):
    cdef int i
    cdef layer_t first_layer = network.lay[0]
    # Fill the firts layer neuron by neuron
    for i in range(first_layer.num_neu):
        first_layer.neu[i].actv = inputs[i]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef network_t train(network_t network, double[:, :] inputs, double [:,:] labels,
                     double[:,:] validation, double[:,:] labels_validation, int epoch, int val_data):
    cdef int i, j, k
    cdef int n_data = len(inputs)
    cdef int n_val_data = len(validation)
    cdef double inv_n_data = 1/<double>n_data
    cdef double inv_n_val = 1/<double>n_val_data
    cdef double error # Temporary error for each train sample
    # Initialize array of errors
    network.train_errors = <double*>malloc(epoch*sizeof(double))
    network.val_errors = <double*>malloc(epoch*sizeof(double))
    for i in range(epoch): # for each epoch
        printf("Epoch %d\r", i)
        for j in range(n_data): # for each data
            feed_input(network, inputs[j])
            forward_prop(network)
            error += compute_error(network, labels[j])
            back_prop(network, labels[j])
        # Batch mode: update after I see all the dataset
        update_weights(network)
        network.train_errors[i] = error*inv_n_data
        error = 0
        if val_data:
            for k in range(n_val_data):
                feed_input(network, validation[k])
                forward_prop(network)
                error += compute_error(network, labels_validation[k])
            network.val_errors[i] = error*inv_n_val
            error = 0
    printf("Done!       \r")
    printf("\n")
    return network

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double[:,:] predict_out(network_t network, double[:,:] data):
    cdef int num_layers = network.num_layers
    cdef int i, j
    cdef int n_data = len(data)
    cdef int n_out_features = network.lay[num_layers-1].num_neu
    # Initialize array of predictions
    cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] predictions = np.empty((n_data, n_out_features))
    for i in range(n_data):
        feed_input(network, data[i]) # Send in the first layer the data
        forward_prop(network)        # Propagate it in the net
        for j in range(n_out_features): # Save the result in predictions
            predictions[i, j] = network.lay[num_layers-1].neu[j].actv
    return predictions

cdef void forward_prop(network_t network):
    cdef int i, j, k
    cdef int num_layers = network.num_layers
    cdef double xw
    cdef layer_t lay_here, lay_prev
    for i in range(1, num_layers): # for each layer
        lay_here = network.lay[i] # store actual layer (just for compat code)
        lay_prev = network.lay[i-1] # store previous layer (just for compact code)
        for j in range(lay_here.num_neu): # for each neuron of acutual layer
            lay_here.neu[j].net = lay_here.neu[j].bias # First add bias to the net
            for k in range(lay_prev.num_neu): # For each neuron of previus layer
                # Select the w of the connection j of the neuron k (prevoius
                # layer) and multiply it by the out of that neuron (wk * xk)
                xw = lay_prev.neu[k].out_weights[j] * lay_prev.neu[k].actv
                # implement sum_k(wk*xk) relative to neuron j (actual layer)
                lay_here.neu[j].net += xw
            # Update the output of the actual neuron with the activation
            # function on the net
            lay_here.neu[j].actv = lay_here.activation_f(lay_here.neu[j].net)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void back_prop(network_t network, double [:] labels):
    cdef int i, j, k, rev
    cdef int num_layers = network.num_layers
    cdef layer_t lay = network.lay[num_layers-1]
    cdef layer_t lay_back = network.lay[num_layers-2]

    ### Last Layer ###
    for j in range(lay.num_neu): # For each neuron of last layer
        # Compute the delta: (d-o)*f'(net)
        lay.neu[j].dnet = (labels[j] - lay.neu[j].actv) * lay.derivative_f(lay.neu[j].net)
        for k in range(lay_back.num_neu): # for each neuron of prev. layer
            # Compute dw_back: delta*input = delta*out_prev
            lay_back.neu[k].dw[j] += lay.neu[j].dnet * lay_back.neu[k].actv
            # Store the value w_prev * delta for next steps
            lay_back.neu[k].dactv += lay_back.neu[k].out_weights[j] * lay.neu[j].dnet
        # Compute dbias just like delta * x0_prev = delta
        lay.neu[j].dbias += lay.neu[j].dnet
    ##################

    ### Inner Layers ###
    for i in range(network.num_layers-2):
        # rev goes from num_layers - 2 to 1
        rev = num_layers - 2 - i # Reversed range does not exist in cython
        lay = network.lay[rev] # actual layer
        lay_back = network.lay[rev-1] # previous layer
        for j in range(lay.num_neu): # for each neu in actual layer
            # KEY OF ALGORITHM: compute next delta based on the previous values
            # w_here * delta_backprop * f'(net)
            # (delta_backprop * w_here) will be updated below in the loop of k
            lay.neu[j].dnet = lay.neu[j].dactv * lay.derivative_f(lay.neu[j].net)
            for k in range(lay_back.num_neu): # for each neuron in prev. lay.
                # Update the dw_back : delta*out_prev
                lay_back.neu[k].dw[j] += lay_back.neu[k].actv * lay.neu[j].dnet
                if rev > 1: # if after this layer there isn't the imputs
                    # Store the value of delta * w_prev for the next step
                    lay_back.neu[k].dactv += lay_back.neu[k].out_weights[j] * lay.neu[j].dnet
            # Compute dbias just like delta * x0_prev (= delta * 1 = delta )
            lay.neu[j].dbias += lay.neu[j].dnet
            lay.neu[j].dactv = 0. # reset the sum(delta * w) to zero (for next training)
    ####################

cdef void update_weights(network_t network):
    cdef int i, j, k
    cdef layer_t layer
    for i in range(network.num_layers-1): # for each layer (not the last)
        lay = network.lay[i] # store the layer (compact code)
        for j in range(lay.num_neu): # for each neuron in the layer
            for k in range(network.lay[i+1].num_neu): # for neu in the next lay
                # update the w_k of neuron j adding eta * dw_k (repeated for
                # each k)
                lay.neu[j].out_weights[k] += network.eta * lay.neu[j].dw[k] \
                                             - 2 * network.l * lay.neu[j].out_weights[k]
                # restore dw (for online mode)
                lay.neu[j].dw[k] = 0.
            # update the bias like eta*dbias_j
            lay.neu[j].bias += network.eta * lay.neu[j].dbias \
                               - 2 * network.l * lay.neu[j].bias
            # restore dbias (for online mode)
            lay.neu[j].dbias = 0.
    # last layer
    lay = network.lay[network.num_layers-1]
    for j in range(lay.num_neu): # for each neuron in the last layer
        # update only the bias (out_weights are not needed)
        lay.neu[j].bias += network.eta * lay.neu[j].dbias \
                           - 2 * network.l * lay.neu[j].bias
        # restore dbias (for online mode)
        lay.neu[j].dbias = 0.

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double compute_error(network_t network, double[:] labels):
    cdef double error_single, sum_error = 0, quadratic_error
    cdef int i, num_layers = network.num_layers
    for i in range(network.lay[num_layers-1].num_neu):
        error_single = labels[i] - network.lay[num_layers-1].neu[i].actv
        quadratic_error = error_single*error_single
        sum_error += quadratic_error
    return sum_error

