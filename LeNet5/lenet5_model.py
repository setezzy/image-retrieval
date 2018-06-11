import tensorflow as tf
# import numpy as np


def weight_init(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    convolution operation

    Args:
      x: image input. 4-d tensor of [batch, height, weight, channel]
      W: filter. 4-d tensor of [height,weight,num_in,num_out]
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x):
    """
    max_pooling operation

    Args:
      x: image input. 4-d tensor
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


def conv_layer(x, weight, bias):
    """
    convolutional layer

    Args:
      x: input layer
      weight: the weight of filter
      bias: bias

    Return:
      h_pool: the feture map after convolution and max pooling
    """
    W_conv = weight_init(weight)
    b_conv = bias_init(bias)

    h_conv = tf.nn.relu(conv2d(x, W_conv) + b_conv)
    h_pool = max_pool(h_conv)

    return h_pool


def fc_layer(x, weight, bias):
    """
    fully-connected layer

    Args:
      x: input layer
      weight: weight of fully-connected layer
      bias: bias of fully-connected layer

    Return:
      h_fc: output of the layer
    """
    W_fc = weight_init(weight)
    b_fc = weight_init(bias)

    h_fc = tf.nn.relu(tf.matmul(x, W_fc) + b_fc)

    return h_fc


def model(x, keep_prob):
    """
    construt the LeNet-5 layers

    Args:
      x: input
      keep_prob: drop out rate

    Return:
      l_: probability
    """

    # input layer
    # padding the input image to 32*32
    x_image = tf.pad(tf.reshape(x, [-1, 28, 28, 1]), [[0, 0], [2, 2], [2, 2], [0, 0]])

    # Layer 1
    h1 = conv_layer(x_image, [5, 5, 1, 6], [6])

    # Layer 2
    h2 = conv_layer(h1, [5, 5, 6, 16], [16])

    # Layer 3
    # without pooling
    weight = weight_init([5, 5, 16, 120])
    bias = bias_init([120])
    h3 = tf.nn.relu(conv2d(h2, weight) + bias)
    h3_flat = tf.reshape(h3, [-1, 120])

    # Layer 4
    # fully-connected layer
    h4 = fc_layer(h3_flat, [120, 84], [84])

    # Layer 5
    # output layer,fully-connected
    # l_ : probability vector
    h4_drop = tf.nn.dropout(h4, keep_prob)
    weight = weight_init([84, 10])
    bias = bias_init([10])
    l_ = tf.nn.softmax(tf.matmul(h4_drop, weight) + bias)

    return l_


