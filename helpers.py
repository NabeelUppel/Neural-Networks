from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases

install_aliases()

import os
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images, test_labels


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def check_my_network(layer_sizes, network):
    correct = True
    if not len(network) + 1 == len(layer_sizes):
        correct = False
        print("The network does not have the correct number of layers")
        print("Number of layers in network: ", len(network))
        print("Expected number of layers: ", len(layer_sizes))
    for i in range(len(network)):
        if not network[i][0].shape == (layer_sizes[i], layer_sizes[i + 1]):
            correct = False
            print("Layer " + str(i) + " does not have the correct dimension weights")
            print("Layer weight shape: ", str(network[i][0].shape))
            print("Expected shape: ", str((layer_sizes[i], layer_sizes[i + 1])))
            print("################################################################")
        if not network[i][1].shape[0] == layer_sizes[i + 1]:
            correct = False
            print("Layer " + str(i) + " does not have the correct dimension biases")
            print("Layer bias shape: ", str(network[i][1].shape[0]))
            print("Expected shape: ", str(layer_sizes[i + 1]))
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    if correct:
        print("The network is correct")
