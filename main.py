import numpy
import numpy as np
import helpers
import math
from Data import DataSetup


def rand_weights_bias(neuronsFirst, neuronsSecond):
    weight_matrix = np.random.rand(neuronsFirst, neuronsSecond)
    bias_vector = np.random.rand(neuronsSecond, 1).flatten()
    return weight_matrix, bias_vector


def initialise(layerArray):
    output = []
    length = len(layerArray)
    for i in range(length + 1):
        if i + 2 <= length:
            x = layerArray[i:i + 2]
            y = rand_weights_bias(x[0], x[1])
            output.append(y)
    return output


training_data, training_labels, test_data, test_labels = helpers.load_mnist()


def sigmoid(z):
    s = 1 / (1 + math.exp(-z))
    return s


def propagate(network, dataPoint):
    layers = len(network)
    vFunct = np.vectorize(sigmoid)
    Z = []
    A = []

    dp = np.insert(dataPoint,0,1,axis=0)
    for i in range(layers):
        if i != layers - 1:
            w = network[i][0]
            b = network[i][1]
            z1 = np.matmul(w, dp)
            z = numpy.add(z1, b)
            a = vFunct(z)
            dp= a
            Z.append(z)
            A.append(a)
    return A, Z


def main():
    trainingData = DataSetup(100).Data
    labels = trainingData[:, 2]
    trainingData = trainingData[:, :2]

    network = initialise([2, 3, 4, 1])  # EXCLUDES BIAS NODES

    a, b = propagate(network, trainingData[0])
    print(a)
    print(b)


if __name__ == "__main__":
    main()
