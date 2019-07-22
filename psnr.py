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
    img_arr1 = numpy.array(im1).astype("float32")
    img_arr2 = numpy.array(im2).astype("float32")
    mse = tf.reduce_mean(tf.squared_difference(img_arr1, img_arr2))
    result = tf.constant(255 ** 2, dtype=tf.float32) / mse
    result = tf.constant(10, dtype=tf.float32) * log10(result)
    with tf.Session():
        result = result.eval()
    return result


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
