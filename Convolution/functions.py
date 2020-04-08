import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def forward_convolution(input_shape, conv_filter_shape, conv_stride, pool_size, pool_stride, activation_fn='relu'):
    """
Creates a model and adds all of the convolution layers
    :param activation_fn:
    :param input_shape:     shape of the input dataset (height, width, nr_channels)
    :param conv_filter_shape: list of tuples representing filter shapes (filter_size, nr_filter_channels) for each layer
    :param conv_stride:       list of integers representing strides for each convolution layer
    :param pool_size:         list of integers representing pooling filter size
    :param pool_stride:       list of integers representing the stride for each pooling layer
    """

    model = models.Sequential()

    n = len(conv_filter_shape)

    model.add(layers.Conv2D(
        filters=conv_filter_shape[0][1],
        kernel_size=conv_filter_shape[0][0],
        strides=conv_stride[0],
        activation=activation_fn,
        input_shape=input_shape
    ))

    model.add(layers.MaxPool2D(
        pool_size=pool_size[0],
        strides=pool_stride[0]
    ))

    for i in range(1,n):
        model.add(layers.Conv2D(
            filters=conv_filter_shape[i][1],
            kernel_size=conv_filter_shape[i][0],
            strides=conv_stride[i],
            activation=activation_fn
        ))
        model.add(layers.MaxPool2D(
            pool_size=pool_size[i],
            strides=pool_stride[i]
        ))

    return model


def forward_fully_connected(model, nr_nodes, activation_fn='relu'):
    n= len(nr_nodes)

    model.add(layers.Flatten())

    for i in range(n-1):
        model.add(layers.Dense(
            units= nr_nodes[i],
            activation=activation_fn,
            #kernel_initializer= tf.keras.initializers.GlorotUniform()
        ))

    model.add(layers.Dense(
        units=nr_nodes[n-1],
        activation=activation_fn,
        #kernel_initializer=tf.keras.initializers.GlorotUniform()
    ))

    return model

