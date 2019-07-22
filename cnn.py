"""
Some description
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
        the deserialized array,
    """
    infile = open(filename, "rb")
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def psnr(im1, im2):
    """
    https://github.com/XiaoCode-er/python-PSNR/blob/master/psnr.py
    """
    mse = tf.reduce_mean(tf.squared_difference(im1, im2))
    result = (
        tf.constant(255 ** 2, dtype=tf.float32) / mse
    )  # TODO: Maybe fix pixel scale later?
    result = tf.math.multiply(tf.constant(10, dtype=tf.float32), log10(result))
    return result.eval()


def generate_file_list(file_path):
    result = []
    for filename in os.listdir(file_path):
        if not filename.startswith("."):
            result.append(filename)
    return result


def gaussian_noise(shape, mean, std):
    return tf.random_normal(shape=shape, mean=mean, stddev=std, dtype=tf.float32)


# TODO: extract [hyper]-parameters, add docstrings and type hints.
PATH = "./dataset/ground/"
IMAGES = generate_file_list(PATH)
DIMENSION = 256
EPOCHS = 10000
LR = 0.001  # TODO: diminish LR as training progresses
BATCH_SIZE = 1


def cnn_model_fn(inputs):
    """
    Inspired by:
    https://www.tensorflow.org/tutorials/estimators/cnn
    Model function for CNN.
    """
    # Input Layer
    input_layer = tf.reshape(inputs, [BATCH_SIZE, DIMENSION, DIMENSION, 1])
    # First Outer Convolutional Layer
    current = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
    )
    # Builds the fifteen inner layers.
    for _ in range(15):
        current = tf.layers.conv2d(
            inputs=current, filters=64, kernel_size=(3, 3), padding="same"
        )
        current = tf.layers.batch_normalization(
            inputs=current, axis=3, momentum=0.0, epsilon=0.0001
        )
        current = tf.nn.relu(current)
    # Second Outer Convolutional Layer
    conv_out_2 = tf.layers.conv2d(
        inputs=current, filters=1, kernel_size=[3, 3], padding="same"
    )
    return conv_out_2


def main():
    # Load in our custom training data -- based on these instructions:
    # https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
    # Evidently, it is not the optimal approach.

    original = tf.read_file(PATH + random.choice(IMAGES))
    original = tf.image.decode_jpeg(original, channels=1, dct_method="INTEGER_ACCURATE")
    original = tf.cast(original, tf.float32)

    noise = gaussian_noise(tf.shape(original), 0, 3)
    noisy_image = original + noise

    output = cnn_model_fn(noisy_image)

    # Calculates real noise.
    flattened_output = tf.layers.flatten(output)
    flattened_noise = tf.reshape(noise, [-1, DIMENSION * DIMENSION])

    # Calculate loss by comparing pixel differences.
    loss = tf.losses.mean_squared_error(
        labels=flattened_noise, predictions=flattened_output
    )

    # Configure the Training Op
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # sess.run(tf.global_variables_initializer())
        # loss_array = []
        saver.restore(sess, "./model/model.ckpt")
        loss_array = load_loss_array()

        for step in range(EPOCHS):

            sess.run(train_op)

            if step % 10 == 0:

                output_noise = tf.squeeze(tf.cast(output, tf.uint8), axis=0)
                output_noise = tf.image.encode_jpeg(
                    output_noise, quality=100, format="grayscale"
                )
                writer = tf.write_file(
                    "./outputs/generated_noise_" + str(step) + ".png", output_noise
                )
                sess.run(writer)

                denoised_image = tf.squeeze(
                    tf.cast(noisy_image - output, tf.uint8), axis=0
                )
                denoised_image = tf.image.encode_jpeg(
                    denoised_image, quality=100, format="grayscale"
                )
                writer = tf.write_file(
                    "./outputs/denoised_image_" + str(step) + ".png", denoised_image
                )

                sess.run(writer)
                current_loss = loss.eval()

                print(
                    "Step "
                    + str(step)
                    + ", Minibatch Loss = "
                    + "{:.4f}".format(loss.eval())
                    + ", PSNR = "
                    + "{:.4f}".format(
                        psnr(tf.squeeze(original), tf.squeeze(noisy_image - output))
                    )
                    + ", Brightest Pixel = "
                    + "{:.4f}".format(tf.reduce_max(output).eval())
                )
                loss_array.append(current_loss)
                saver.save(sess, "./model/model.ckpt")
                save_loss_array(loss_array)


if __name__ == "__main__":
    main()
