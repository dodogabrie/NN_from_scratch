# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np # Make numpy work with cython
from libc.stdlib cimport malloc, free
cimport topology
cimport training
from topology cimport network_t, create_network, create_neuron
from training cimport feed_input, forward_prop, back_prop, train, predict_out, reserve_inputs
from generate_neurons cimport find_best_candidate

ctypedef np.double_t DOUBLE_t
ctypedef long INT_t  # np.int_t no longer exists in numpy 2

cdef class network:
    cdef network_t net

    def __init__(self, int num_inputs, int num_outputs, str out_activation = "sigmoid",
                 str hidden_activation = "sigmoid", double eta = 0.1, double w_init = 0.1):
        self.net = create_network(num_inputs, num_outputs, out_activation,
                                  hidden_activation, eta, w_init)

    @property
    def eta(self):
        return self.net.eta

    @property
    def num_hidden(self):
        return self.net.num_hidden

    def get_weights(self):
        """Per neuron (hidden first, then outputs): (weights, bias). Hidden i sees inputs + hidden < i."""
        cdef int i, j
        out = []
        for i in range(self.net.tot_neurons):
            w = np.array([self.net.neu[i].w_in[j] for j in range(self.net.neu[i].num_w_in)])
            out.append((w, self.net.neu[i].bias))
        return out

    def train(self, inputs_raw, labels_raw, int epoch, int patience = 50, int max_hidden = 0,
              double err_solved = 0.01, double max_plateau = 1e-4,
              val_raw = None, labels_val_raw = None, val_dataset = False,
              callback = None, int pool = 8, int cand_iter = 300, double cand_rate = 1.):
        """Cascade Correlation: train the output units; every `patience` epochs, if the error
        improved less than `max_plateau`, add the best of `pool` candidates (max covariance with
        the residual error), freeze it and go on. callback(net, epochs_done) is called after each step."""
        cdef network_t net = self.net # extract the net
        cdef int n_out = net.num_outputs # Get neu in last lay
        cdef int n_in = net.num_inputs # Get neu in first lay
        cdef int done = 0, n
        cdef double prev, cur
        # Reformatting Labels of inputs as matrix
        labels_raw = np.array(labels_raw, dtype = np.double)
        labels_raw = np.reshape(labels_raw, (len(labels_raw), n_out))
        cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] labels = labels_raw
        # Reformatting inputs as matrix
        inputs_raw = np.array(inputs_raw, dtype = np.double)
        inputs_raw = np.reshape(inputs_raw, (len(inputs_raw), n_in))
        cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] inputs = inputs_raw

        if val_dataset:
            labels_val_raw = np.array(labels_val_raw, dtype = np.double)
            labels_val_raw = np.reshape(labels_val_raw, (len(labels_val_raw), n_out))
            val_raw = np.array(val_raw, dtype = np.double)
            val_raw = np.reshape(val_raw, (len(val_raw), n_in))
        else:
            labels_val_raw = labels_raw
            val_raw = inputs_raw
        cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] labels_val = labels_val_raw
        cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] val = val_raw
        reserve_inputs(&net, n_in + max_hidden)
        net.train_errors = <double*>malloc(epoch*sizeof(double))
        net.val_errors = <double*>malloc(epoch*sizeof(double))
        while done < epoch:
            n = min(patience, epoch - done)
            net = train(net, inputs, labels, done, n, val, labels_val, val_dataset)
            prev = net.train_errors[done - 1] if done > 0 else 1e9
            done += n
            cur = net.train_errors[done - 1]
            self.net = net
            if callback is not None: callback(self, done)
            if cur < err_solved: break
            if prev - cur < max_plateau and net.num_hidden < max_hidden:
                net = find_best_candidate(net, inputs, labels, pool, cand_iter, cand_rate)
                self.net = net
                if callback is not None: callback(self, done)
        for n in range(done, epoch):  # epochs not run keep the last error
            net.train_errors[n] = net.train_errors[done - 1]
            net.val_errors[n] = net.val_errors[done - 1]
        self.net = net

    def predict(self, double[:,:] data):
        return np.array(predict_out(self.net, data))

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_train_error(self, int epoch, int val = 0):
        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=1, mode='c'] errors = np.empty(epoch)
        cdef np.ndarray[DOUBLE_t, ndim=1, mode='c'] val_errors = np.empty(epoch)
        for i in range(epoch):
            errors[i] = self.net.train_errors[i]
            if val: val_errors[i] = self.net.val_errors[i]
        if val:
            return errors, val_errors
        else:
            return errors
