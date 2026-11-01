# distutils: language=c++
"""Cascade Correlation candidate stage: train a pool of candidate neurons to maximise the
covariance between their output and the residual error, then add the best one to the net."""
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free, realloc
from libc.math cimport fabs
from topology cimport neuron_t, network_t, create_neuron
from training cimport feed_input, forward_prop

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)     # Make division fast like C
cdef network_t find_best_candidate(network_t net, double[:,:] data, double[:,:] labels,
                         int pool_size, int n_iter, double rate):
    cdef int c, it, i, j, o, n = len(data), d = net.num_inputs + net.num_hidden, n_out = net.num_outputs
    cdef double[:,:] X = np.empty((n, d))      # what every candidate sees, per sample
    cdef double[:,:] E = np.empty((n, n_out))  # residual error of the outputs, mean removed
    cdef double[:] V = np.empty(n), dV = np.empty(n), cov = np.empty(n_out), Ebar = np.zeros(n_out)
    cdef double[:] g = np.empty(d)
    cdef double Vbar, gb, S, S_best = -1.
    cdef neuron_t cand, best
    for i in range(n):
        feed_input(net, data[i])
        forward_prop(net)
        for j in range(d): X[i, j] = net.inputs[j]
        for o in range(n_out):
            E[i, o] = labels[i, o] - net.neu[net.num_hidden + o].actv
            Ebar[o] += E[i, o] / n
    for i in range(n):
        for o in range(n_out): E[i, o] -= Ebar[o]
    for c in range(pool_size):
        cand = generate_candidate(net)
        for it in range(n_iter + 1):
            Vbar = 0.
            for i in range(n):
                cand.net = cand.bias
                for j in range(d): cand.net += cand.w_in[j] * X[i, j]
                V[i] = cand.actv_f(cand.net)
                dV[i] = cand.actv_df(cand.net)
                Vbar += V[i] / n
            S = 0.
            for o in range(n_out):
                cov[o] = 0.
                for i in range(n): cov[o] += (V[i] - Vbar) * E[i, o]
                S += fabs(cov[o])
            if it == n_iter: break
            # gradient ascent on S = sum_o |cov_o|
            for j in range(d): g[j] = 0.
            gb = 0.
            for o in range(n_out):
                for i in range(n):
                    gb += (1. if cov[o] >= 0 else -1.) * E[i, o] * dV[i]
                    for j in range(d): g[j] += (1. if cov[o] >= 0 else -1.) * E[i, o] * dV[i] * X[i, j]
            for j in range(d): cand.w_in[j] += rate * g[j] / n
            cand.bias += rate * gb / n
        print('candidate', c, 'S =', S)
        if S > S_best:
            if S_best >= 0: free(best.w_in); free(best.dw_in)
            S_best = S
            best = cand
        else:
            free(cand.w_in); free(cand.dw_in)
    return add_neuron(net, best)

cdef neuron_t generate_candidate(network_t net):
    return create_neuron(net.num_inputs + net.num_hidden, net.w_init, net.hidden_actv_f, net.hidden_actv_df)

cdef network_t add_neuron(network_t net, neuron_t neu):
    # New hidden neuron goes after the old ones (before the outputs); every output gets one more weight
    cdef int i, nh = net.num_hidden
    net.neu = <neuron_t*>realloc(net.neu, (net.tot_neurons + 1) * sizeof(neuron_t))
    for i in range(net.tot_neurons - 1, nh - 1, -1): net.neu[i + 1] = net.neu[i]
    neu.inputs = net.inputs
    net.neu[nh] = neu
    for i in range(nh + 1, net.tot_neurons + 1):
        net.neu[i].w_in = <double*>realloc(net.neu[i].w_in, (net.neu[i].num_w_in + 1) * sizeof(double))
        net.neu[i].dw_in = <double*>realloc(net.neu[i].dw_in, (net.neu[i].num_w_in + 1) * sizeof(double))
        net.neu[i].w_in[net.neu[i].num_w_in] = 0.
        net.neu[i].dw_in[net.neu[i].num_w_in] = 0.
        net.neu[i].num_w_in += 1
    net.num_hidden += 1
    net.tot_neurons += 1
    return net
