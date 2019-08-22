"""
Inspired by: 

"""

import tensorflow as tf


def gen_cnn_model_fn(inputs):
    """ 

    Args:
        input: data passed to the input layer of the neural network -- in our case, it
        is the noisy image.
    Returns:
        the output of the neural network -- in our case, it represents the predicted
        noise in the image.
    """
    # Sets batch size to one, if it is only a single image.
    if len(inputs.get_shape()) < 4:
        inputs = tf.expand_dims(inputs, 0)
    with tf.variable_scope("generator"):
        current = inputs
        # Standard Convolutional Layers:
        for _ in range(3):
            current = tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                kernel_initializer="Orthogonal",
                padding="same",
            )(current)
            current = tf.keras.layers.BatchNormalization(
                axis=3, momentum=0.0, epsilon=0.0001
            )(current)
            current = tf.nn.leaky_relu(current)
        # Residual Layers:
        for _ in range(3):
            inputs = current
            for _ in range(2):
                current = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    kernel_initializer="Orthogonal",
                    padding="same",
                )(current)
                current = tf.keras.layers.BatchNormalization(
                    axis=3, momentum=0.0, epsilon=0.0001
                )(current)
                current = tf.nn.leaky_relu(current)
            current = tf.identity(inputs) + current
    # Final Convolutional Layer:
    output = tf.keras.layers.Conv2D(
        filters=3, kernel_size=[3, 3], kernel_initializer="Orthogonal", padding="same"
    )(current)
    return output


def get_dis_conv(layers, name, filters, strides, kernel):
    """ Reusing...
    """
    if not name in layers:
        padding = "same"
        if name == "output_layer_conv":
            padding = "valid"
        temp = tf.keras.layers.Conv2D(
            filters=filters,
            strides=strides,
            kernel_size=kernel,
            kernel_initializer="Orthogonal",
            padding=padding,
        )
        layers[name] = temp
    return layers[name]


def get_dis_bn(layers, name):
    """ Reusing...
    """
    if not name in layers:
        temp = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001)
        layers[name] = temp
    return layers[name]


def dis_cnn_model_fn(inputs, layers):
    """
    binary classifier!
    """
    with tf.variable_scope("discriminator"):
        # First Outer Convolutional Layer:
        current = get_dis_conv(layers, "input_layer_conv", 48, 2, [4, 4])(inputs)
        current = get_dis_bn(layers, "input_layer_bn")(current)
        current = tf.nn.relu(current)
        # First Inner Layer:
        current = get_dis_conv(layers, "conv_1", 96, 2, [4, 4])(current)
        current = get_dis_bn(layers, "bn_1")(current)
        current = tf.nn.relu(current)
        # Second Inner Layer:
        current = get_dis_conv(layers, "conv_2", 192, 2, [4, 4])(current)
        current = get_dis_bn(layers, "bn_2")(current)
        current = tf.nn.relu(current)
        # Third Inner Layer:
        current = get_dis_conv(layers, "conv_3", 384, 2, [4, 4])(current)
        current = get_dis_bn(layers, "bn_3")(current)
        current = tf.nn.relu(current)
        # Second Outer Convolutional Layer:
        current = get_dis_conv(layers, "output_layer_conv", 1, 1, [4, 4])(current)
        current = get_dis_bn(layers, "output_layer_bn")(current)
    return tf.squeeze(tf.nn.sigmoid(current))
