""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from click.core import batch
mnist = input_data.read_data_sets("c:/tmp/data/", one_hot=True)
batch_size = 50
batch_x, batch_y = mnist.train.next_batch(batch_size)
_col = 5
_row = int(batch_size / _col)
_, (ax)= plt.subplots(_row,_col)
for i in range(0, _row):
    for j in range(0, _col):
        print(i*_col + j)
        if (i*_col + j) < batch_size:
            image = batch_x[i*_col + j]
            image = np.frombuffer(image, dtype=np.uint8)
            ax[i][j].imshow(image.reshape(28, 28, 4), cmap=plt.cm.get_cmap());
plt.show()