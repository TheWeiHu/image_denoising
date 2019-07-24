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

# TODO: test on original images (probably seperate file)


def save_loss_array(table, filename="model/loss_array"):
    """An array tracking the loss over epochs is serialized using pickle and dumped
    into an output file.

    Args:
        table: an array tracking the loss over epochs.
        filename: the name of the file into which the serialized data is dumped.
    """
    outfile = open(filename, "wb")
    pickle.dump(table, outfile)
    outfile.close()


def load_loss_array(filename="model/loss_array"):
    """Deserializes a an array tracking the loss over epochs.

    Args:
        filename: the name of the file from which the data is to be deserialized.
    Returns:
        the deserialized array.
    """
    infile = open(filename, "rb")
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict


def log10(real_number):
    """Calculate the base-ten log of a given real number.
    Args:
        real_number: a real number.
    Returns:
        the base-ten log of the given real number.
    """
    numerator = tf.log(real_number)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    """Calculates the peak signal-to-noise ratio between two images. This quantitative
    measure serves as a heuristic for evaluating similarity.
    The algorithm is inspired by:
    https://github.com/XiaoCode-er/python-PSNR/blob/master/psnr.py

    Args:
        im1: the first image.
        im1: the second image.
    Returns:
        the peak signal-to-noise ratio.
    """
    mse = tf.reduce_mean(tf.squared_difference(im1, im2))
    result = tf.constant(1, dtype=tf.float32) / mse
    result = tf.math.multiply(tf.constant(10, dtype=tf.float32), log10(result))
    return result


def generate_file_list(file_path):
    """Generates a list of all the files in a given directory.

    Args:
        file_path: the directory whose files are of interest.
    Returns:
        an array containing the names of all the files in the given directory.
    """
    result = []
    for filename in os.listdir(file_path):
        # Ignores irrelevant files.
        if not filename.startswith("."):
            result.append(filename)
    return result


def gaussian_noise(shape, mean, std):
    """Takes i.i.d. samples from a Gaussian distribution with the given parameters,
    which serves as noise to be added to an image.

    Args:
        shape: the dimension of the i.i.d. samples.
        mean: the mean of the Gaussian distribution
        std: the standard deviation of the Gaussian distribution -- the larger this
        value, the noisier the produced image is said to be.
    Returns:
        a tensor of the given shape which serves as Gaussian noise.
    """
    return tf.random_normal(shape=shape, mean=mean, stddev=std, dtype=tf.float32)


# Algorithm Hyperparameters:
PATCH_PATH = "./dataset/patch/"
IMAGE_PATH = "./dataset/ground/"
IMAGES = generate_file_list(IMAGE_PATH)
PATCHES = generate_file_list(PATCH_PATH)
DIMENSION = 64
EPOCHS = 10000
SCALE = 255.0  # Note the use of a float to prevent integer division.
LR = 0.001  # Updated using Adam Optimizer
BATCH_SIZE = 128
STDV = 25  # The standard deviation used for the gaussian noise.


def cnn_model_fn(inputs, batch_size=BATCH_SIZE):
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
    # Input Layer
    input_layer = tf.reshape(inputs, [batch_size, DIMENSION, DIMENSION, 1])
    # First Outer Convolutional Layer:
    current = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        kernel_initializer="Orthogonal",
        padding="same",
        activation=tf.nn.relu,
    )
    # Builds the fifteen inner layers.
    for _ in range(15):
        current = tf.layers.conv2d(
            inputs=current,
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer="Orthogonal",
            padding="same",
        )
        current = tf.layers.batch_normalization(
            inputs=current, axis=3, momentum=0.0, epsilon=0.0001
        )
        current = tf.nn.relu(current)
    # Second Outer Convolutional Layer:
    conv_out_2 = tf.layers.conv2d(
        inputs=current,
        filters=1,
        kernel_size=[3, 3],
        kernel_initializer="Orthogonal",
        padding="same",
    )
    return conv_out_2


def train(noise, input_images, original, noisy_image, output):
    """
    Description
    """
    # Calculate loss by comparing pixel differences.
    loss = tf.losses.mean_squared_error(labels=noise, predictions=output)
    # Configure the Training Op
    train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(
        loss=loss, global_step=tf.train.get_global_step()
    )

    with tf.Session() as sess:

        # Creates new model or restores previouslt saved model.
        # sess.run(tf.global_variables_initializer())
        # loss_array = []
        tf.train.Saver().restore(sess, "./model/model.ckpt")
        loss_array = load_loss_array()

        for step in range(EPOCHS):

            # Randomly selects a ground image.
            ground_patches = [
                PATCH_PATH + random.choice(PATCHES) for _ in range(BATCH_SIZE)
            ]
            # Runs training process.
            _, _loss = sess.run(
                [train_op, loss],
                feed_dict={i: d for i, d in zip(input_images, ground_patches)},
            )
            loss_array.append(_loss)

            if step % 10 == 0:
                # Log the current training status.
                _psnr, _max, _min = sess.run(
                    [
                        psnr(tf.squeeze(original), tf.squeeze(noisy_image - output)),
                        tf.reduce_max(output),
                        tf.reduce_min(output),
                    ],
                    feed_dict={i: d for i, d in zip(input_images, ground_patches)},
                )
                print(
                    "Step "
                    + str(step)
                    + ", Minibatch Loss = "
                    + "{:.8f}".format(_loss)
                    + ", PSNR = "
                    + "{:.4f}".format(_psnr)
                    + ", Brightest Pixel = "
                    + "{:.4f}".format(_max * SCALE)
                    + ", Darkest Pixel = "
                    + "{:.4f}".format(_min * SCALE)
                )

                # Serialize trained network and the progression of loss values.
                tf.train.Saver().save(sess, "./model/model.ckpt")
                save_loss_array(loss_array)


def evaluate():
    """
    Description
    """
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, "./model/model.ckpt")


def main():
    """ Implements the deep convolutional neural network for denoising images based on
    the paper by Zhang et al.
    """
    # Loads in batch of randonly chosen patches.
    input_images = [tf.placeholder(tf.string) for _ in range(BATCH_SIZE)]

    # Loads in and "concatenates" the selected patches.
    original = []
    for patch in input_images:
        current = tf.read_file(patch)
        current = tf.image.decode_jpeg(
            current, channels=1, dct_method="INTEGER_ACCURATE"
        )
        current = tf.math.divide(tf.cast(current, tf.float32), SCALE)
        original.append(current)
    original = tf.stack(original)

    # Generates Gaussian noise and adds it to the image.
    noise = tf.math.divide(gaussian_noise(tf.shape(original), 0, STDV), SCALE)
    noisy_image = original + noise

    # Inputs noisy image into the neural network.
    output = cnn_model_fn(noisy_image)

    # Trains the model.
    train(noise, input_images, original, noisy_image, output)


if __name__ == "__main__":
    main()
