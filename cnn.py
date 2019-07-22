"""
Some description
"""

import os

# TODO: eliminate use of cv2
import cv2

# import numpy as np
import tensorflow as tf
from psnr import psnr

# TODO: extract [hyper]-parameters, add docstrings and type hints.
DIMENSION = 256
EPOCHS = 10000
LR = 0.001  # TODO: diminish LR as training progresses
BATCH_SIZE = 1


def generate_file_list(file_path):
    result = []
    # TODO: handle more than one image.
    for filename in os.listdir(file_path):
        if not filename.startswith(".") and filename.endswith("01.png"):
            result.append(filename)
    return result


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

    file_list = generate_file_list("./dataset/noisy")
    labels = []
    for i in file_list:
        labels.append(i[-6:])
    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def _parse_function(filename, label):
        image_string = tf.read_file("./dataset/noisy/" + filename)
        image_decoded = tf.image.decode_jpeg(
            image_string, channels=1, dct_method="INTEGER_ACCURATE"
        )
        image = tf.cast(image_decoded, tf.float32)
        return filename + "___" + label, image, label

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(BATCH_SIZE)
    # A "one-shot" iterator does not support re-initialization.
    iterator = dataset.make_one_shot_iterator()
    F, images, labels = iterator.get_next()

    output = cnn_model_fn(images)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "./model/model.ckpt")

        for step in range(EPOCHS):
            # Read in the original image corresponding with the label.
            original = tf.read_file(
                "./dataset/ground/" + labels.eval()[0].decode("utf-8")
            )
            original = tf.image.decode_jpeg(
                original, channels=1, dct_method="INTEGER_ACCURATE"
            )
            # Calculates real noise.
            flattened_output = tf.layers.flatten(output)
            flattened_original = tf.reshape(original, [-1, DIMENSION * DIMENSION])
            flattened_image = tf.reshape(images, [-1, DIMENSION * DIMENSION])
            flattened_original = tf.cast(flattened_original, tf.float32)
            flattened_image = tf.cast(flattened_image, tf.float32)
            real_noise = tf.math.subtract(flattened_original, flattened_image)

            # Calculate loss by comparing pixel differences.
            loss = tf.losses.mean_squared_error(
                labels=real_noise, predictions=flattened_output
            )

            # Configure the Training Op
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step()
            )

            sess.run(train_op)

            if step % 10 == 0:
                # Preserves generated noise.
                cv2.imwrite(
                    "./outputs/generated_noise_" + str(step) + ".png",
                    tf.squeeze(output).eval(),
                )
                # Preserves denoised image.
                cv2.imwrite(
                    "./outputs/denoised_image_" + str(step) + ".png",
                    tf.squeeze(images - output).eval(),
                )

                print(
                    "Step "
                    + str(step)
                    + ", Minibatch Loss= "
                    + "{:.4f}".format(loss.eval())
                    + ", PSNR = "
                    + "{:.4f}".format(
                        psnr(
                            tf.squeeze(original).eval(),
                            tf.squeeze(images - output).eval(),
                        )
                    )
                    + ", Brightest Pixel = "
                    + "{:.4f}".format(tf.reduce_max(output).eval())
                )
                saver.save(sess, "./model/model.ckpt")


if __name__ == "__main__":
    main()
