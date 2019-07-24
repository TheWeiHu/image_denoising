"""
Implementation of a deep convolutional neural network for denoising images based on the
paper by Zhang et al. [https://www.ncbi.nlm.nih.gov/pubmed/28166495]

The authors' original implementation can be found here [https://github.com/cszn/DnCNN].

This implentation serves as a baseline against which our proposed method will be
evaluated.

Isak Persson
Wei Hu

22 July 2019
"""

import os
import pickle
import random
import tensorflow as tf

from cnn_train import psnr, generate_file_list, gaussian_noise



def generate_noise(path_to_image, output, step):
    """ Preserve a visual representation of the generated noise.
    """
    output_noise = tf.squeeze(
        tf.cast(tf.math.multiply(output, SCALE), tf.uint8), axis=0
    )
    output_noise = tf.image.encode_jpeg(output_noise, quality=100, format="grayscale")
    writer = tf.write_file(
        "./outputs/generated_noise_" + str(step) + ".png", output_noise
    )
    return writer


def generate_denoised_image(output, noisy_image, step):
    """ Preserve a visual representation of the denoised image.
    """
    denoised_image = tf.squeeze(
        tf.cast(tf.math.multiply(noisy_image - output, SCALE), tf.uint8), axis=0
    )
    denoised_image = tf.image.encode_jpeg(
        denoised_image, quality=100, format="grayscale"
    )
    writer = tf.write_file(
        "./outputs/denoised_image_" + str(step) + ".png", denoised_image
    )
    return writer
