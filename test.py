import numpy as np
import time
import matplotlib.pyplot as plt
import NN

def test():
    N = int(1e2)

    start = time.time()
    structure = np.array([2, 3, 4, 1], dtype = np.int32)
    inputs = [[0,0],
              [1,0],
              [1,1]]
    activations = np.array(["sigmoid"]*len(structure), dtype=str)
    network = NN.network(structure, activations, 0.1)
    network.feed_input(np.array(inputs, dtype=float), 0)
    network.forward_prop()
    network.back_prop(np.array([1.]))
    print(f'Elapsed {time.time() - start} seconds')
    print(network.num_layers)

if __name__ == '__main__':
    test()
