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

import argparse
import os
import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf


def save_loss_array(table, filename="model/loss_array"):
    """ An array tracking the loss over epochs is serialized using pickle and dumped
    into an output file.

    Args:
        table: an array tracking the loss over epochs.
        filename: the name of the file into which the serialized data is dumped.
    """
    outfile = open(filename, "wb")
    pickle.dump(table, outfile)
    outfile.close()


def load_loss_array(filename="model/loss_array"):
    """ Deserializes a an array tracking the loss over epochs.

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
    """ Calculate the base-ten log of a given real number.
    Args:
        real_number: a real number.
    Returns:
        the base-ten log of the given real number.
    """
    numerator = tf.math.log(real_number)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    """ Calculates the peak signal-to-noise ratio between two images. This quantitative
    measure serves as a heuristic for evaluating similarity.
    The algorithm is inspired by:
    https://github.com/XiaoCode-er/python-PSNR/blob/master/psnr.py

    Args:
        im1: the first image.
        im1: the second image.
    Returns:
        the peak signal-to-noise ratio.
    """
    mse = tf.reduce_mean(tf.math.squared_difference(im1, im2))
    result = tf.constant(1, dtype=tf.float32) / mse
    result = tf.math.multiply(tf.constant(10, dtype=tf.float32), log10(result))
    return result


def generate_file_list(file_path):
    """ Generates a list of all the files in a given directory.

    Args:
        file_path: the directory whose files are of interest.
    Returns:
        an array containing the names of all the files in the given directory.
    """
    result = []
    for filename in os.listdir(file_path):
        # Ignores irrelevant files.
        if filename.endswith(".png"):
            result.append(file_path + filename)
    random.shuffle(result)
    return result


def preprocess_image(image):
    """ https://www.tensorflow.org/tutorials/load_data/images
    """
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.math.divide(
        tf.cast(image, tf.float32), SCALE
    )  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    """ https://www.tensorflow.org/tutorials/load_data/images
    """
    image = tf.io.read_file(path)
    return preprocess_image(image)


def gaussian_noise(shape, mean, std):
    """ Takes i.i.d. samples from a Gaussian distribution with the given parameters,
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


# Program Parameters:
PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--evaluate",
    dest="train",
    action="store_false",
    help="evaluation mode (attempts to denoise images from the test set)",
)
# TODO: Set up both grayscale and color mode.
PARSER.set_defaults(train=True)
ARGS = PARSER.parse_args()

# Algorithm Hyperparameters
PATCH_PATH = "./dataset/patch/"
TEST_PATH = "./dataset/test/"
TESTS = generate_file_list(TEST_PATH)
PATCHES = generate_file_list(PATCH_PATH)
EPOCHS = 10000
SCALE = 255.0  # Note the use of a float to prevent integer division.
LR = 0.001  # Updated using Adam Optimizer
BATCH_SIZE = 128
STDV = 25  # The standard deviation used for the gaussian noise.
N_EVAL = 20  # Number of test to run per evaluation.


def create_dataset_iterator(pathes):
    """ Given a list of paths.... 

    Args:
        pathes: ...
    Returns:
        an iterator...
    """
    path_ds = tf.data.Dataset.from_tensor_slices(pathes)
    image_ds = path_ds.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = image_ds.cache()  # Especially performant if the data fits in memory.
    dataset = dataset.shuffle(buffer_size=len(pathes))
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset.make_one_shot_iterator()


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
    current = tf.layers.conv2d(
        inputs=inputs,
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
        filters=3,
        kernel_size=[3, 3],
        kernel_initializer="Orthogonal",
        padding="same",
    )
    return conv_out_2


def train(loss, original, noisy_image, output):
    """ The main procedure for training the model. On every tenth step, statistics about
    the how the training is progressing is printed out.

    Args:
        loss: the loss function which the model is aiming to minimize.
        original: the original image which is composed of a batch of patches.
        noisy_image: the original image with Gaussian noise added.
        output: the noise generated by the neural network.
    """
    # Configures the Training Op
    train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=LR).minimize(
        loss=loss, global_step=tf.train.get_global_step()
    )

    with tf.Session() as sess:

        # Creates new model or restores previouslt saved model.
        tf.compat.v1.train.Saver().restore(sess, "./model/model.ckpt")
        loss_array = load_loss_array()
        # sess.run(tf.global_variables_initializer())
        # loss_array = []

        for step in range(EPOCHS):

            # Runs training process.
            _, _loss = sess.run([train_op, loss])
            loss_array.append(_loss)

            if step % 10 == 0:
                # Log the current training status.
                _psnr, _max, _min = sess.run(
                    [
                        psnr(tf.squeeze(original), tf.squeeze(noisy_image - output)),
                        tf.reduce_max(output),
                        tf.reduce_min(output),
                    ]
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


def evaluate(loss, input_image, original, noisy_image, output):
    """ A random test image is selected (only a portion of which the network has seen).
    Noise is added to the image which is then fed to the network. We preserve the
    generated noisy image, the noise the network produced, and the denoised image. We
    also output the loss and the PSNR.

    Args:
        loss: the loss function which the model is aiming to minimize.
        input_image: a placeholder which will be fed the path of the test image.
        original: the original test image.
        noisy_image: the original image with Gaussian noise added.
        output: the noise generated by the neural network.
    """
    with tf.Session() as sess:

        tf.train.Saver().restore(sess, "./model/model.ckpt")

        sample_image = random.choice(TESTS)

        _psnr, _loss = sess.run(
            [psnr(tf.squeeze(original), tf.squeeze(noisy_image - output)), loss],
            feed_dict={input_image: sample_image},
        )
        print("Loss = " + "{:.8f}".format(_loss) + ", PSNR = " + "{:.4f}".format(_psnr))

        # Preserves the output of the cnn (the generated noise).
        output_noise = tf.squeeze(
            tf.cast(tf.math.multiply(output, SCALE), tf.uint8), axis=0
        )
        output_noise = tf.image.encode_jpeg(output_noise, quality=100)
        noise_writer = tf.write_file(
            "./outputs/generated_noise_" + sample_image[:7] + ".png", output_noise
        )

        # Preserves the denoised image (the noisy image minus the generated noise).
        denoised_image = tf.squeeze(
            tf.cast(tf.math.multiply(noisy_image - output, SCALE), tf.uint8), axis=0
        )
        denoised_image = tf.image.encode_jpeg(denoised_image, quality=100)
        denoise_writer = tf.write_file(
            "./outputs/denoised_image_" + sample_image[:7] + ".png", denoised_image
        )

        # Preserves the original noisy image.
        _noisy_image = tf.cast(tf.math.multiply(noisy_image, SCALE), tf.uint8)
        _noisy_image = tf.image.encode_jpeg(_noisy_image, quality=100)
        noisy_image_writer = tf.write_file(
            "./outputs/noisy_image_" + sample_image[:7] + ".png", _noisy_image
        )

        sess.run(
            [noisy_image_writer, noise_writer, denoise_writer],
            feed_dict={input_image: sample_image},
        )


def main():
    """ Implements the deep convolutional neural network for denoising images based on
    the paper by Zhang et al.

    There are two modes. During training mode, the model takes in a batch of 64 by 64
    image patches from the training set. During evaluation mode, the the model attempts
    to denoise images of varying sizes from the test set.
    """
    if not ARGS.train:
        # TODO: Use iterator instead of feeding placeholder, once we figure out how to
        # get of the image given by the iterator.
        input_image = tf.placeholder(tf.string)
        original = load_and_preprocess_image(input_image)
    else:
        iterator = create_dataset_iterator(PATCHES)
        original = iterator.get_next()

    # Generates Gaussian noise and adds it to the image.
    noise = tf.math.divide(gaussian_noise(tf.shape(original), 0, STDV), SCALE)
    noisy_image = original + noise

    # Inputs noisy image into the neural network.
    if not ARGS.train:
        output = cnn_model_fn(noisy_image)
        noise = tf.expand_dims(noise, 0)
    else:
        output = cnn_model_fn(noisy_image)

    # Calculates loss by comparing pixel-wise differences.
    loss = tf.losses.mean_squared_error(labels=noise, predictions=output)

    # Trains the model.
    if not ARGS.train:
        plt.plot(load_loss_array())
        plt.savefig("./outputs/loss_progression.pdf")
        for _ in range(N_EVAL):
            evaluate(loss, input_image, original, noisy_image, output)
    else:
        train(loss, original, noisy_image, output)


if __name__ == "__main__":
    main()
