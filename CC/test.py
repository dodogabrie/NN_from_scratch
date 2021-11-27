import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import CC

def test_and():
    epoch = 1000

    inputs = [[0,0],
              [1,0],
              [0,1],
              [1,1]]
    labels = [0, 0, 0, 1]
    network = CC.network(2, 1, eta = 1, w_init = 0.5)
    network.train(inputs, labels, epoch)

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

if __name__ == '__main__':
    test_and()

