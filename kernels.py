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
    """
    Taken from:
    https://github.com/OctoberChang/MMD-GAN/blob/master/mmd.py
    """
    K_XX, K_XY, K_YY, const_diagonal = _mix_rbf_kernel(X, Y)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal, biased)


def _mix_rbf_kernel(X, Y, wts=None, K_XY_only=False):
    """
    Taken from:
    https://github.com/mbinkowski/MMD-GAN/blob/master/gan/core/mmd.py
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


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    """
    Taken from:
    https://github.com/OctoberChang/MMD-GAN/blob/master/mmd.py
    """
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        return (
            tf.reduce_sum(K_XX) / (m * m)
            + tf.reduce_sum(K_YY) / (n * n)
            - 2 * tf.reduce_sum(K_XY) / (m * n)
        )
    if const_diagonal is not False:
        const_diagonal = tf.cast(const_diagonal, tf.float32)
        trace_X = m * const_diagonal
        trace_Y = n * const_diagonal
    else:
        trace_X = tf.trace(K_XX)
        trace_Y = tf.trace(K_YY)
    return (
        (tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
        + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
        - 2 * tf.reduce_sum(K_XY) / (m * n)
    )


def one_sided(inputs):
    """
    Taken from:
    https://github.com/OctoberChang/MMD-GAN/blob/master/mmd.py
    """
    outputs = tf.nn.relu(-inputs)
    return -tf.reduce_mean(outputs)
