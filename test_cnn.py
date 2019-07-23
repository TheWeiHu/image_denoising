### NOT YET IMPLEMENTED!


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
