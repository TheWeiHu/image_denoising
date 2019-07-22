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
PATH = "./dataset/ground/"
IMAGES = generate_file_list(PATH)
DIMENSION = 256
EPOCHS = 10000
SCALE = 255.0  # Note the use of a float to prevent integer division.
LR = 0.001  # TODO: diminish LR as training progresses?
BATCH_SIZE = 1
STDV = 25  # The standard deviation used for the gaussian noise.


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
    # Input Layer
    input_layer = tf.reshape(inputs, [BATCH_SIZE, DIMENSION, DIMENSION, 1])
    # First Outer Convolutional Layer
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
    # Second Outer Convolutional Layer
    conv_out_2 = tf.layers.conv2d(
        inputs=current,
        filters=1,
        kernel_size=[3, 3],
        kernel_initializer="Orthogonal",
        padding="same",
    )
    return conv_out_2


def main():
    """ Implements the deep convolutional neural network for denoising images based on
    the paper by Zhang et al.
    """
    # Loads in randomly chosen ground image.
    path = tf.placeholder(tf.string)
    original = tf.read_file(path)
    original = tf.image.decode_jpeg(original, channels=1, dct_method="INTEGER_ACCURATE")
    original = tf.math.divide(tf.cast(original, tf.float32), SCALE)

    # Generates Gaussian noise and adds it to the image.
    noise = tf.math.divide(gaussian_noise(tf.shape(original), 0, STDV), SCALE)
    noisy_image = original + noise

    # Inputs noisy image into the neural network.
    output = cnn_model_fn(noisy_image)
    # Calculate loss by comparing pixel differences.
    loss = tf.losses.mean_squared_error(
        labels=tf.reshape(noise, [-1, DIMENSION * DIMENSION]),
        predictions=tf.layers.flatten(output),
    )
    # Configure the Training Op
    train_op = tf.train.GradientDescentOptimizer(learning_rate=LR).minimize(
        loss=loss, global_step=tf.train.get_global_step()
    )

    with tf.Session() as sess:

        # Creates new model or restores previouslt saved model.
        # sess.run(tf.global_variables_initializer())
        # loss_array = []
        tf.train.Saver().restore(sess, "./model/model.ckpt")
        loss_array = load_loss_array()

        for step in range(EPOCHS):

            # Random ly selects a ground image.
            ground_image = PATH + random.choice(IMAGES)
            # Runs training process.
            sess.run(train_op, feed_dict={path: ground_image})

            if step % 10 == 0:
                # Calculate the current loss -- which is to be preserved in array
                # (to be later used for graphing).
                current_loss = loss.eval(feed_dict={path: ground_image})
                loss_array.append(current_loss)
                # Log the current training status.
                print(
                    "Step "
                    + str(step)
                    + ", Minibatch Loss = "
                    + "{:.8f}".format(loss.eval(feed_dict={path: ground_image}))
                    + ", PSNR = "
                    + "{:.4f}".format(
                        psnr(
                            tf.squeeze(original), tf.squeeze(noisy_image - output)
                        ).eval(feed_dict={path: ground_image})
                    )
                    + ", Brightest Pixel = "
                    + "{:.4f}".format(
                        tf.reduce_max(output).eval(feed_dict={path: ground_image})
                        * SCALE
                    )
                )

                # Serialize trained network and the progression of loss values.
                tf.train.Saver().save(sess, "./model/model.ckpt")
                save_loss_array(loss_array)

            if step % 100 == 0:
                # Preserve a visual representation of the generated noise.
                output_noise = tf.squeeze(
                    tf.cast(tf.math.multiply(output, SCALE), tf.uint8), axis=0
                )
                output_noise = tf.image.encode_jpeg(
                    output_noise, quality=100, format="grayscale"
                )
                writer = tf.write_file(
                    "./outputs/generated_noise_" + str(step) + ".png", output_noise
                )
                sess.run(writer, feed_dict={path: ground_image})
                # Preserve a visual representation of the denoised image.
                denoised_image = tf.squeeze(
                    tf.cast(tf.math.multiply(noisy_image - output, SCALE), tf.uint8),
                    axis=0,
                )
                denoised_image = tf.image.encode_jpeg(
                    denoised_image, quality=100, format="grayscale"
                )
                writer = tf.write_file(
                    "./outputs/denoised_image_" + str(step) + ".png", denoised_image
                )
                sess.run(writer, feed_dict={path: ground_image})


if __name__ == "__main__":
    main()
