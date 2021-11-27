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
from training cimport feed_input, forward_prop

ctypedef np.double_t DOUBLE_t

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef network_t find_best_candidate(network_t net, double[:,:] data, double[:,:] labels,
                         int pool_size, double tollerance):
    cdef int i, j, n_data = len(data)
    cdef double S_candidate = 0, S_candidate_prev = tollerance + 1, S_best = 0
    cdef neuron_t candidate
    cdef neuron_t best_candidate
    for i in range(pool_size):
        print('candidate', i)
        candidate = generate_candidate(net)
        while abs(S_candidate - S_candidate_prev) > tollerance:
            print('S before maximization')
            S_candidate = compute_S(net, candidate, data, labels)
            print(S_candidate)
            candidate = update_candidate_w(net, candidate, data, labels)
            S_candidate_prev = S_candidate
            print('S after maximization')
            S_candidate = compute_S(net, candidate, data, labels)
            print(S_candidate)
            print('\n')
        S_candidate_prev = S_candidate + 2*tollerance
#            print(S_candidate)
#        if S_candidate > S_best:
#            S_best = S_candidate
#            best_candidate = candidate
#    net = add_space_for_neuron(net)
#    net.neu[net.num_hidden] = best_candidate
#    net.num_hidden += 1
#    net.tot_neurons += 1
    return net

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef neuron_t update_candidate_w(network_t net, neuron_t new_neu, double[:,:] data, double[:,:] labels):
    cdef int i, n_data = len(data)
    cdef double S_j
#    print('try to update w new neu')
    for i in range(net.num_outputs):
#        print('computing the means for out', i)
        E_mean = compute_mean_error(net, i, data, labels)
        O_mean = compute_mean_out(net, new_neu, i, data, labels)
        S_j = 0
        new_neu.dnet = 0
#        print('starting loop on data')
        for j in range(n_data):
#            print('feed_input')
            feed_input(net, data[j])
#            print('forward_prop')
            forward_prop(net)
#            print('Connect the new neu')
            new_neu = manually_connect_neu(net, new_neu)
            O_ij = net.neu[net.num_hidden + i].actv
            Err_ij = labels[i, j] - O_ij
            S_j += (new_neu.actv - O_mean) * (Err_ij - E_mean)
            new_neu.dnet += (Err_ij - E_mean) * new_neu.actv_df(new_neu.net)
        if S_j >= 0: S_j = 1.
        else: S_j = -1
        for k in range(new_neu.num_w_in):
            print('incoming inputs:', new_neu.inputs[k])
            new_neu.dw_in[k] += S_j * new_neu.dnet * new_neu.inputs[k]
    for i in range(new_neu.num_w_in):
        print('dw', i, '=', new_neu.dw_in[i])
        new_neu.w_in[i] += net.eta * new_neu.dw_in[i]
    return new_neu

cdef neuron_t generate_candidate(network_t net):
    cdef neuron_t neu_c = create_neuron(net.num_inputs + net.num_hidden,
                                        net.w_init, net.hidden_actv_f, net.hidden_actv_df)
    return neu_c

cdef neuron_t manually_connect_neu(network_t net, neuron_t new_neu):
    cdef int i
    cdef double tmp
#    print('add bias')
    new_neu.net = new_neu.bias
#    print('add inputs')
    for i in range(net.num_inputs):
#        print('input', i, 'found')
#        print(new_neu.inputs[i],net.inputs[i])
        new_neu.inputs[i] = net.inputs[i]
#        print(new_neu.inputs[i],net.inputs[i])
#        print('adding to net')
        new_neu.net += net.inputs[i] * new_neu.w_in[i]
#    print('add from hidden')
    for i in range(net.num_hidden):
        new_neu.inputs[net.num_inputs + i] = net.neu[i].actv
        new_neu.net += net.neu[i].actv * new_neu.w_in[net.num_inputs + i]
#    print('compute activation func')
    new_neu.actv = new_neu.actv_f(new_neu.net)
    return new_neu


cdef network_t add_space_for_neuron(network_t net):
    cdef int i
    cdef double * tmp_w = NULL
    cdef neuron_t * tmp_neu = NULL

    tmp_neu = <neuron_t*>realloc(net.neu, (net.tot_neurons + 1) * sizeof(neuron_t))
    if tmp_neu == NULL:
        print("Failed reallocation!")
    else:
        for i in range(net.tot_neurons + 1):
            if i < net.num_hidden:     tmp_neu[i] = net.neu[i]
            if i > net.num_hidden:     tmp_neu[i] = net.neu[i-1]
        net.neu = tmp_neu

    for i in range(net.num_outputs):
        tmp_w = <double*>realloc(net.neu[i].w_in, (net.neu[i].num_w_in + 1) * sizeof(double))
        if tmp_w == NULL:
              print("Failed reallocation!")
        else:
            tmp_w[net.neu[i].num_w_in] = net.w_init * (1 - rand()/RAND_MAX)
            net.neu[i].w_in = tmp_w
            net.neu[i].num_w_in += 1
    return net


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double compute_S(network_t net, neuron_t new_neu, double[:,:] data, double[:,:] labels):
    cdef double S_j = 0, S=0, O_ij, Err_ij
    cdef int i, j, n_data = len(data)
    for i in range(net.num_outputs):
        E_mean = compute_mean_error(net, i, data, labels)
        O_mean = compute_mean_out(net, new_neu, i, data, labels)
        for j in range(n_data):
            feed_input(net, data[j])
            forward_prop(net)
            new_neu = manually_connect_neu(net, new_neu)
            O_ij = net.neu[net.num_hidden + i].actv
            Err_ij = labels[i, j] - O_ij
            S_j += (O_ij - O_mean) * (Err_ij - E_mean)
        S += abs(S_j)
        S_j = 0
    return S

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double compute_mean_error(network_t net, int j, double[:,:] data, double[:,:] labels):
    cdef int i
    cdef int n_data = len(data)
    cdef double E = 0, out
    for i in range(n_data):
        feed_input(net, data[i])
        forward_prop(net)
        out = net.neu[net.num_hidden + j].actv
        E += out - labels[i, j]
    return E/n_data

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double compute_mean_out(network_t net, neuron_t new_neu, int j, double[:,:] data, double[:,:] labels):
    cdef int i
    cdef int n_data = len(data)
    cdef double mean_out = 0
#    print('compute_mean_out')
    for i in range(n_data):
#        print('feed_input')
        feed_input(net, data[i])
#        print('forward_prop')
        forward_prop(net)
#        print('manually_connect_neu')
        new_neu = manually_connect_neu(net, new_neu)
#        print('algebra')
        out = new_neu.actv
        mean_out += out
    return mean_out/n_data
