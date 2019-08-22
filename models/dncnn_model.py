"""
Deep convolutional neural network model, which takes in a noisy input image, and
attempts to produce an output which represents the noise in the image.
"""

import tensorflow as tf

def cnn_model_fn(inputs):
    """ A deep convolutional neural network with seventeen layers, with batch
    normalization and relu activation. The hyperparameters are those used in the
    authors' implementation.

    Inspired by:
    https://www.tensorflow.org/tutorials/estimators/cnn

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
    # First Outer Convolutional Layer:
    current = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[3, 3],
        kernel_initializer="Orthogonal",
        padding="same",
        activation=tf.nn.relu,
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
        current = tf.nn.relu(current)
    # Second Outer Convolutional Layer:
    conv_out_2 = tf.keras.layers.Conv2D(
        filters=3, kernel_size=[3, 3], kernel_initializer="Orthogonal", padding="same"
    )(current)
    return conv_out_2
