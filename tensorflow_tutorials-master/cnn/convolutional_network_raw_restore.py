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
mnist = input_data.read_data_sets("c:/tmp/data/", one_hot=True)
model_path = 'c:/tmp/tensorflow_model/con_net/'
logs_path = 'c:/tmp/tensorflow_logs/con_net_restore/'

num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
keep_prob = 0.8
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])


def conv_net(x, weights, biases, dropout):
    print("777777777777777777777777777777777777777777777777777")
    
#     showImg(X,28,4)

    sess = tf.Session()
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    a = conv1.eval()
    a = tf.transpose(a)
    tf.summary.image("filtered_images_layer1", a, max_outputs=32)
    a = a.eval()
#     showImg2(a[0],28,4)
#     showImg2(a[1],28,4)
#     showImg2(a[2],28,4)
#     showImg2(a[3],28,4)
#     showImg2(a[4],28,4)
#     showImg2(a[5],28,4)
#     showImg2(a[6],28,4)
#     showImg2(a[7],28,4)
#     showImg2(a[8],28,4)
    print(a)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    a = conv1.eval()
    print(a)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    a = conv2.eval()
    print(a)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    a = conv2.eval()
    print(a)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, 3136])
    a = fc1.eval()
    print(a)
    
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    a = fc1.eval()
    print(a)
    
    fc1 = tf.nn.relu(fc1)
    a = fc1.eval()
    print(a)
    
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    a = fc1.eval()
    print(a)
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    a = out.eval()
    print(a)
    return out

def showImg(x,size,chanle):
    _, (ax)= plt.subplots(1,1)
    image = np.frombuffer(x, dtype=np.uint8)
    image = image.reshape(size, size, chanle)
    ax.imshow(image, cmap=plt.cm.get_cmap());
    plt.show()
    
def showImg2(x,size,chanle):
    _, (ax)= plt.subplots(1,1)
    image = np.frombuffer(x, dtype=np.uint8)
    image = image.reshape(size, size, chanle)
    ax.imshow(image, cmap=plt.cm.get_cmap());
    plt.show()

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

saver = tf.train.import_meta_graph(model_path + "model.ckpt.meta")

merged_summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model_path + "model.ckpt")
    graph = tf.get_default_graph()
    print("restore ok: " , graph)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': sess.run(graph.get_tensor_by_name('weights/wc1:0')),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': sess.run(graph.get_tensor_by_name('weights/wc2:0')),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': sess.run(graph.get_tensor_by_name('weights/wd1:0')),
        # 1024 inputs, 10 outputs (class prediction)
        'out': sess.run(graph.get_tensor_by_name('weights/out:0'))
    }
    
    biases = {
        'bc1': sess.run(graph.get_tensor_by_name('biases/bc1:0')),
        'bc2': sess.run(graph.get_tensor_by_name('biases/bc2:0')),
        'bd1': sess.run(graph.get_tensor_by_name('biases/bd1:0')),
        'out': sess.run(graph.get_tensor_by_name('biases/out:0'))
    }
    
    print("============================================")
    X = mnist.test.images[:1]
    _logistic = conv_net(X, weights, biases, keep_prob)
    

    _,summary = sess.run([_logistic,merged_summary_op], feed_dict={X: mnist.test.images[:1],
                                  Y: mnist.test.labels[:1],
                                  keep_prob: 1.0})
    print(_logistic)
    print("********************************************")

