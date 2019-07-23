import tensorflow as tf
import psnr

img1 = tf.constant(1.0, shape=[224, 256])
img2 = tf.constant(1.0, shape=[3, 3])

result = psnr.psnr(img1, img2)

with tf.Session() as sess:
    print(sess.run(result))



def psnr(im1, im2):
    """
    https://github.com/XiaoCode-er/python-PSNR/blob/master/psnr.py
    """
    mse = tf.reduce_mean(tf.squared_difference(im1, im2))
    result = tf.constant(255 ** 2, dtype=tf.float32) / mse  # TODO: Maybe fix pixel scale later?
    result = tf.math.multiply(tf.constant(10, dtype=tf.float32), log10(result))


    return result.eval()