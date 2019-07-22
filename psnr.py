import os
import numpy
import cv2
import tensorflow as tf


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    """
    https://github.com/XiaoCode-er/python-PSNR/blob/master/psnr.py
    """
    mse = tf.reduce_mean(tf.squared_difference(im1, im2))
    result = tf.constant(255 ** 2, dtype=tf.float32) / mse  # TODO: Maybe fix pixel scale later?
    result = tf.math.multiply(tf.constant(10, dtype=tf.float32), log10(result))
    return result.eval()


def main():
    for filename in os.listdir("./dataset/ground"):
        if not filename.startswith("."):
            for i in range(1000):
                original = cv2.imread(
                    "./dataset/ground/" + filename, cv2.IMREAD_GRAYSCALE
                )
                contrast = cv2.imread(
                    "./dataset/noisy/noisy_" + str(i) + "_" + filename,
                    cv2.IMREAD_GRAYSCALE,
                )
                print(psnr(original, contrast))


if __name__ == "__main__":
    main()
