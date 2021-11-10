import numpy as np
import sys
sys.path.append('NN/')
import time
import matplotlib.pyplot as plt
import NN

def test():
#    N = int(1e2)
#
#    structure = np.array([3, 2, 1], dtype = np.int32)
#    inputs = [[0,0,0],
#              [1,0,0],
#              [0,1,0],
#              [0,0,1],
#              [1,1,0],
#              [1,0,1],
#              [0,1,1],
#              [1,1,1]]
#    labels = [0, 1, 1, 1, 1, 1, 1, 0]
#    inputs = np.array(inputs, dtype = float)
#    labels = np.reshape(np.array(labels, dtype = float), (len(inputs), 1))
#    activations = np.array(["sigmoid"]*len(structure), dtype=str)
#    network = NN.network(structure, activations, eta = .2, w_init = 1.)
#    network.print_network()
#    start = time.time()
#    errors = network.train(inputs, labels, 5000)
#    print(f'Elapsed time for training: {time.time() - start} seconds')
#    plt.plot(errors)
#    plt.show()

    data=np.loadtxt("dataset.csv", delimiter=",")
    N_val = int(len(data)/10)

    input_data=data[:-N_val,1:-2]
    labels=data[:-N_val,-2:]

    val_data = data[-N_val:,1:-2]
    val_labels=data[-N_val:,-2:]


    # Building
    epoch = 75
    structure = np.array([input_data.shape[1], 4, 2], dtype = np.int32)
    activations = np.array(['sigmoid', 'sigmoid', 'linear'])
    network = NN.network(structure, activations, eta = .0002, w_init = .1)
    start = time.time()
    network.train(input_data, labels, epoch)
    print(f'Elapsed time for training: {time.time() - start} seconds')
    errors = network.get_train_error(epoch)
    plt.plot(errors)
    plt.show()
    start = time.time()
    pred = network.predict(val_data)
    print(f'Elapsed time for predict: {time.time() - start} seconds')


if __name__ == '__main__':
    test()
