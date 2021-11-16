# distutils: language=c++
import numpy as np
cimport cython
cimport numpy as np # Make numpy work with cython
from libc.stdlib cimport malloc, free
cimport topology
cimport training
from topology cimport network_t, create_network, create_neuron
from training cimport feed_input, forward_prop, back_prop, train, predict_out

ctypedef np.double_t DOUBLE_t
ctypedef np.int_t INT_t

cdef class network:
    cdef network_t net

    def __init__(self, int[:] structure, np.ndarray activations, double eta, double w_init):
        self.net = create_network(structure, activations, eta, w_init)

    @property
    def eta(self):
        return self.net.eta

    @property
    def num_layers(self):
        return self.net.num_layers

    @property
    def neu_in_layer(self):
        cdef int i, n_layers = self.num_layers
        cdef np.ndarray[INT_t, ndim=1, mode='c'] neu_in_layer = np.zeros(n_layers).astype(int)
        for i in range(n_layers):
            neu_in_layer[i] = self.net.lay[i].num_neu
        return neu_in_layer

    def print_network(self):
        for i in range(self.num_layers):
            print('Layer', i, '-->', self.net.lay[i].num_neu, 'neu')

    def train(self, inputs_raw, labels_raw, int epoch):
        cdef network_t net = self.net # extract the net
        cdef int num_layers = self.num_layers # Get number of layers
        cdef int n_out = net.lay[num_layers-1].num_neu # Get neu in last lay
        cdef int n_in = net.lay[0].num_neu # Get neu in first lay
        # Reformatting Labels of inputs as matrix
        labels_raw = np.array(labels_raw, dtype = np.double)
        labels_raw = np.reshape(labels_raw, (len(labels_raw), n_out))
        cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] labels = labels_raw
        # Reformatting inputs as matrix
        inputs_raw = np.array(inputs_raw, dtype = np.double)
        inputs_raw = np.reshape(inputs_raw, (len(inputs_raw), n_in))
        cdef np.ndarray[DOUBLE_t, ndim=2, mode='c'] inputs = inputs_raw
        self.net = train(net, inputs, labels, epoch)


    def predict(self, double[:,:] data):
        return np.array(predict_out(self.net, data))

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def get_train_error(self, int epoch):
        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=1, mode='c'] errors = np.empty(epoch)
        for i in range(epoch):
            errors[i] = self.net.train_errors[i]
        return errors
