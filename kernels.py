"""
Inspired the paper "MMD-GAN: Towards Deeper Understanding of Moment Matching Network"

Taken from the accompagnying code:
https://github.com/OctoberChang/MMD-GAN

Inspired the paper "Demystifying MMD GANs"

Taken from the accompagnying code:
https://github.com/mbinkowski/MMD-GAN
"""

import tensorflow as tf


def mmd2(X, Y, biased=False):
    """ Calculates the MMD loss
    Taken from:
    https://github.com/OctoberChang/MMD-GAN/blob/master/mmd.py
    Args:
        X: Encoded images from the generator with size (batch size, channels)
        Y: Encoded images from the ground truth with size (batch size, channels)
    return:
        The loss based on the MMD
    """
    K_XX, K_XY, K_YY, const_diagonal = _mix_rbf_kernel(X, Y)
    return _mmd2(K_XX, K_XY, K_YY)


def _mix_rbf_kernel(X, Y, wts=None, K_XY_only=False):
    """a Calculates the kernel distance between rows of X and rows of Y. Returns these distances in a matrix
    Taken from:
    https://github.com/mbinkowski/MMD-GAN/blob/master/gan/core/mmd.py

    Args:
        X: Encoded images from the generator with size (batch size, channels).
        Y: Encoded images from the ground truth with size (batch size, channels).
        wts: Weights for the sigmas.
        K_XY_only = If you only want to return the K_XY matrix
    return:
        K_XX: Kernel distances between rows in X
        K_XY: Kernel distances between rows in X and rows in Y
        K_YY: Kernel distances between rows in Y
        tf.reduce_sum(wts): sum of the weights
    """
    sigmas = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    if wts is None:
        wts = [1] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0

    XYsqnorm = -2 * XY + c(X_sqnorms) + r(Y_sqnorms)
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XY += wt * tf.exp(-gamma * XYsqnorm)

    if K_XY_only:
        return K_XY

    XXsqnorm = -2 * XX + c(X_sqnorms) + r(X_sqnorms)
    YYsqnorm = -2 * YY + c(Y_sqnorms) + r(Y_sqnorms)

    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XX += wt * tf.exp(-gamma * XXsqnorm)
        K_YY += wt * tf.exp(-gamma * YYsqnorm)

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)


def _mmd2(K_XX, K_XY, K_YY):
    """ Sums up the kernal distances to create the MMD loss
    Taken from:
    https://github.com/OctoberChang/MMD-GAN/blob/master/mmd.py
    Args:
        K_XX: Kernel distances between rows in X
        K_XY: Kernel distances between rows in X and rows in Y
        K_YY: Kernel distances between rows in Y
    return:
        MMD loss

    """
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    trace_X = tf.trace(K_XX)
    trace_Y = tf.trace(K_YY)

    return (
        (tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
        + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
        - 2 * tf.reduce_sum(K_XY) / (m * n)
    )