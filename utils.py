"""
A collection of general helper functions.
"""

import argparse
import os
import random
from PIL import Image
import image_slicer
import tensorflow as tf


def scale(image):
    """ Normalizes each pixel value to be in the [0,1] range.

    Args:
        image: the original image.
    Returns:
        the original image normalized.
    """
    image = tf.cast(image, tf.float32)
    image /= 255
    return image


def unscale(image):
    """ Restores an image whose pixel values have been normalized to be in the [0,1]
    range.

    Args:
        image: the normalized image
    Returns:
        the original image.
    """
    return tf.cast(tf.math.multiply(image, 255), tf.uint8)


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

    def log10(real_number):
        """ Calculate the base-ten log of a given real number.

        Args:
            real_number: a real number.
        Returns:
            the base-ten log of the given real number.
        """
        numerator = tf.math.log(real_number)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

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
        # Only accepts PNG images.
        if filename.endswith(".png"):
            result.append(file_path + filename)
    random.shuffle(result)
    return result


def preprocess_image(image):
    """ Preprocesses an image file which has been read into memory.
    From: https://www.tensorflow.org/tutorials/load_data/images

    Args:
        path: an image file read into memory.
    Returns:
        a tensor representation of the image.
    """
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_ACCURATE")
    return scale(image)


def load_and_preprocess_image(path):
    """ Loads and preprocesses the image found at the given path.
    From: https://www.tensorflow.org/tutorials/load_data/images

    Args:
        path: a path to an image.
    Returns:
        a tensor representation of the image.
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
    return tf.random.normal(shape=shape, mean=mean, stddev=std, dtype=tf.float32)


def create_dataset_iterator(pathes, batch_size=64):
    """ Given a list of paths to images, the corresponding images are loaded into memory
    and shuffled. An iterator is created which repeatively loops over the images (when
    the iterator has looped through all the images, the images are shuffled, and the
    process is repeated).

    Args:
        pathes: a list of pathes to images.
    Returns:
        an iterator which continously supply batches of the images.
    """
    path_ds = tf.data.Dataset.from_tensor_slices(pathes)
    image_ds = path_ds.map(
        load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = image_ds.cache()  # Especially performant if the data fits in memory.
    dataset = dataset.shuffle(buffer_size=len(pathes))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return tf.compat.v1.data.make_one_shot_iterator(dataset)


def write_image(path, image):
    """ Create a node which will write the given image to file, when evaluated.

    Args:
        path: the location where the image will be saved.
    Returns:
        a graph node which will write to file when evaluated.
    """
    image = tf.image.encode_jpeg(image, quality=100)
    return tf.io.write_file(path, image)

def create_summary(image_summaries, scalar_summaries):
    """ Creates summaries to be shown on TensorBoard.

    Args:
        image_summaries: a dictionary whose keys are labels and values are images.
        scalar_summaries: a dictionary whose keys are labels and values are scalars.
    Returns:
        The summaries merged.
    """
    for key, value in image_summaries.items():
        tf.summary.image(key, unscale(value))
    for key, value in scalar_summaries.items():
        tf.summary.scalar(key, value)
    return tf.summary.merge_all()


def get_args():
    """ Defines the arguments accepted by the program.

    Returns:
        The arguments given to the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        dest="train",
        action="store_false",
        help="Evaluation Mode (attempts to denoise images from the test set).",
    )
    parser.add_argument(
        "--initialize",
        dest="initialize",
        action="store_true",
        help="Initializes the Graph (instead of loading from previous checkpoint).",
    )
    parser.set_defaults(train=True)
    parser.set_defaults(initialize=False)
    return parser.parse_args()


def make_tiles(input_path, save_path, dimension):
    """Given a square image and a dimension, the image will be cut into tiles of that
    dimension, whenever possible.

    Args:
        input_path: the file path which lead to the image which is to be cut up.
        save_path: the directory in which the cut up pieces will be stored.
        dimension: the prescribed dimension of the cut up tiles.
    """
    for filename in os.listdir(input_path):
        if filename.endswith(".png"):
            image_path = input_path + filename

            width, height = Image.open(image_path).size

            # Ensures image is square.
            assert width == height
            # Ensures the image can be cut into the desired dimensions.
            assert width % dimension == 0
            n_tiles = (width / dimension) ** 2

            tiles = image_slicer.slice(image_path, n_tiles, save=False)
            image_slicer.save_tiles(
                tiles, directory=save_path, prefix=filename[0:2], format="png"
            )


# Hyperparameters
PATCH_PATH = "./dataset/patch/"
TEST_PATH = "./dataset/test/"
TESTS = generate_file_list(TEST_PATH)
PATCHES = generate_file_list(PATCH_PATH)

if __name__ == "__main__":
    make_tiles("./dataset/train/", "./dataset/patch/", 64)
