"""
A generator and an encoder/decoder pair which compose a generative adverserial network
that is trained using kernel maximum mean discrepency used for denoising images.
"""


import tensorflow as tf
import numpy as np


def gen_cnn_model_fn(inputs):
    """ A deep convolutional neural network with seventeen layers, with batch
    normalization and relu activation. The hyperparameters are those used in the
    authors' implementation.

    NOTE: this is the essentially the same model as the one used in the DnCNN, except,
    we return "inputs - output" instead of just "output"! This returns the denoised
    image instead of the generated noise.

    Inspired by:
    https://www.tensorflow.org/tutorials/estimators/cnn

    Args:
        input: data passed to the input layer of the neural network -- in our case, it
        is the noisy image.
    Returns:
        the output of the neural network -- for us, it represents the denoised image.
    """
    # Sets batch size to one, if it is only a single image.
    if len(inputs.get_shape()) < 4:
        inputs = tf.expand_dims(inputs, 0)
    with tf.variable_scope("generator"):
        # First Outer Convolutional Layer:
        current = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            kernel_initializer="Orthogonal",
            padding="same",
            activation=tf.nn.leaky_relu,
        )(inputs)
        # Builds the fifteen inner layers.
        for _ in range(15):
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
        # Second Outer Convolutional Layer:
        current = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=[3, 3],
            kernel_initializer="Orthogonal",
            padding="same",
        )(current)
    return inputs - current


def d_decoder(inputs, batch_size=64, size=64, reuse=True):
    """
    :param x: (batch size, 1, 1, ???)
    :return: (batch size, 64, 64, 3)
    """

    with tf.variable_scope("d_decoder") as scope:
        if reuse:
            scope.reuse_variables()

        current = tf.reshape(
            inputs, shape=[batch_size, 1, 1, 128]
        )  # TODO: the shape should be an argument of the function.

        i = np.log2(size) - 1
        while i >= 0:
            current = tf.layers.conv2d_transpose(
                current,
                filters=4 * 2 ** i,
                kernel_size=4,
                kernel_initializer="orthogonal",
                strides=2,
                padding="same",
                use_bias=False,
            )
            current = tf.layers.batch_normalization(
                inputs=current, axis=3, momentum=0.0, epsilon=0.0001
            )
            current = tf.nn.relu(current)
            i -= 1

        current = tf.layers.conv2d_transpose(
            current,
            filters=3,
            kernel_size=2,
            kernel_initializer="orthogonal",
            strides=1,
            padding="same",
            use_bias=False,
        )

        output = tf.math.tanh(current)
        return output


def d_encoder(inputs, batch_size=64, size=64, reuse=True):
    """
    Used to encode images from the generator and the ground truth.
    Output will be used as input for the kernels.
    :param x: batch_size * 64 * 64 * channels
    :return: batch_size * 1 * 1 * k
    """

    with tf.variable_scope("d_encoder") as scope:
        if reuse:
            scope.reuse_variables()

        if len(inputs.get_shape()) < 4:
            inputs = tf.expand_dims(inputs, 0)

        current = tf.reshape(
            inputs, shape=[batch_size, size, size, 3]
        )  # TODO: the shape/isize should be an argument of the function.
        assert size % 16 == 0

        i = 0
        while size / 2 ** (i) > 1:
            current = tf.layers.batch_normalization(
                inputs=current, axis=3, momentum=0.0, epsilon=0.0001
            )
            current = tf.layers.conv2d(
                inputs=current,
                filters=4* 2**(i),
                kernel_size=(3, 3),
                kernel_initializer="orthogonal",
                activation=tf.nn.relu,
                padding="same",
            )
            current = tf.layers.max_pooling2d(current, pool_size=(2, 2), strides=(2, 2))
            i += 1
        return current
