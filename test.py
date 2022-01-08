import numpy as np
import sys
sys.path.append('NN/')
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cyNeural as cyNN

def test_xor():
    epoch = 1000

    structure = np.array([2, 2, 1], dtype = np.int32)
    inputs = [[0,0],
              [1,0],
              [0,1],
              [1,1]]
    labels = [0, 1, 1, 0]
    activations = np.array(["sigmoid"]*len(structure), dtype=str)
    network = cyNN.network(structure, activations, eta = 2., w_init = 1.)
    network.print_network()
    start = time.time()
    network.train(inputs, labels, epoch)
    print(f'Elapsed time for training: {time.time() - start} seconds')
    network.get_train_error(epoch)
    errors = network.get_train_error(epoch)
    fig, ax = plt.subplots(1,2, figsize = (10,4))
    ax[0].plot(errors)
    ax[0].set_title('error per epoch')
    x_arr = np.linspace(0,1,100)
    y_arr = np.linspace(0,1,100)
    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)
    for i, x in enumerate(x_arr):
        for j, y in enumerate(y_arr):
            Z[i][j] = network.predict(np.array([[x,y]]))[0][0]
    c = ax[1].pcolormesh(X, Y, Z, shading='auto')
    ax[1].set_title('decision boundary')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(c, cax = cax)
    plt.suptitle('show results')
    plt.show()

def MONK_test():
    train = np.loadtxt('data/monks-1.train', usecols = np.arange(7)).astype(float)
    train_labels = train[:, 0].reshape((len(train), 1))
    test = np.loadtxt('data/monks-1.test', usecols = np.arange(7)).astype(float)
    test_labels = test[:, 0].reshape((len(test), 1))
    epoch = 1000
    structure = np.array([train.shape[1], 3, 1], dtype = np.int32)
    activations = np.array(['sigmoid', 'sigmoid', 'sigmoid'])
    network = cyNN.network(structure, activations, eta = .01, w_init = .1)
    start = time.time()
    network.train(train, train_labels, epoch)
    print(f'Elapsed time for training: {time.time() - start} seconds')
    errors = network.get_train_error(epoch)

    pred = network.predict(test)
    mean_errors_predict = np.sqrt(np.sum((pred-test_labels)**2))/len(test)
    print(mean_errors_predict)

    plt.plot(errors)
    plt.show()
    return

def test_ML_cup():
    data=np.loadtxt("data/dataset.csv", delimiter=",")
    N_val = int(len(data)/10)

    input_data=data[:-N_val,1:-2]
    labels=data[:-N_val,-2:]

    val_data = data[-N_val:,1:-2]
    val_labels=data[-N_val:,-2:]


    # Building
    epoch = 1000
    structure = np.array([input_data.shape[1], 10, 5, 2], dtype = np.int32)
    activations = np.array(['sigmoid', 'sigmoid', 'sigmoid', 'linear'])
    start = time.time()
    network = cyNN.network(structure, activations, eta = .0001, w_init = 0.01, l = 0.0002)
    print(f'Time for initialize the net: {time.time()-start} seconds')
    start = time.time()
    network.train(input_data, labels, epoch)
    print(f'Elapsed time for training: {time.time() - start} seconds')
    errors = network.get_train_error(epoch)
    plt.plot(errors)
    plt.yscale('log')
    plt.show()
    start = time.time()
    pred = network.predict(val_data)
    print(f'Elapsed time for predict: {time.time() - start} seconds')
    e = np.sum((pred-val_labels)**2, axis = 1)
    errors_predict = np.sum(np.sqrt(e))/len(pred)
    print('error on predicted data:', errors_predict)

def ML_data_understanding():
    data=np.loadtxt("data/dataset.csv", delimiter=",")
    input_data=data[:,1:-2]
    labels=data[:,-2:]

#    fig, axs = plt.subplots(input_data.shape[1], 1)
#    for feat, ax in zip(input_data.T, axs.flatten()):
#        ax.boxplot(feat, vert = False)
#    plt.show()
    plt.scatter(input_data[:,0], input_data[:, 2], c = labels[:,0])
    plt.colorbar()
    plt.show()

def ML_validation_check():
    data=np.loadtxt("data/dataset.csv", delimiter=",")
    N_val = int(len(data)/10)

    input_data=data[:-N_val,1:-2]
    labels=data[:-N_val,-2:]

    val_data = data[-N_val:,1:-2]
    val_labels=data[-N_val:,-2:]
    epoch = 1000
    structure = np.array([input_data.shape[1], 4, 2], dtype = np.int32)
    activations = np.array(['sigmoid', 'sigmoid', 'linear'])
    val_err = []
    err = []
    network = cyNN.network(structure, activations, eta = .0001, w_init = .1)
    for i in range(epoch):
        network.train(input_data, labels, epoch = 1)
        err.append(network.get_train_error(1))
        pred = network.predict(val_data)
        val_err.append(np.sqrt(np.sum((pred-val_labels)**2)))
    plt.plot(np.array(err))
    plt.plot(np.array(val_err))
    plt.show()
    return

def test_tick_reg():
    epoch = 10000

    def f(x, w=1, a = 1, noise = True):
        return a*np.sin(w*x) + noise * np.random.rand(len(x)) * a/3

    x = np.linspace(0, 2*np.pi, 200)
    f_eval = f(x)
    inputs = x[::2].reshape((len(x[::2]), 1))
    labels = f_eval[::2].reshape((len(x[::2]), 1))
    val = x[1::2]
    val_label = f(val, noise = False)
    structure = np.array([1, 20, 2, 1], dtype = np.int32)
    activations = np.array(["sigmoid", 'sigmoid', 'sigmoid', 'linear'], dtype=str)
    network = cyNN.network(structure, activations, eta = .004, w_init = 1., l = 0.0004)
    network.print_network()
    start = time.time()
    network.train(inputs, labels, epoch, val, val_label, 1)
    errors, val_errors = network.get_train_error(epoch, val = 1)
    plt.plot(errors, c = 'blue')
    plt.plot(val_errors, c = 'red')
    plt.show()

    val = val.reshape((len(val), 1))
    pred = network.predict(val)
    plt.plot(val, pred, c = 'red')
    plt.scatter(inputs, labels, c = 'blue')
    plt.show()


if __name__ == '__main__':
    test_ML_cup()
