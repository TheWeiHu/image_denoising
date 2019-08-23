"""
A generator and a discriminator which compose a generative adverserial network used for
denoising images.
"""

import tensorflow as tf


def gen_cnn_model_fn(inputs):
    """ A neural network acting as the generator in the generative adverserial network.
    Inspired by U-Net, and ResNet, tt uses, convolutional layers, residual layers, and
    deconvolutional layers to produce a denoised image.

    Inspired by:
    https://github.com/manumathewthomas/ImageDenoisingGAN

    BUT IS BEING MODIIFIED to use a U-NET structure:
    https://arxiv.org/abs/1505.04597

    Args:
        input: data passed to the input layer of the neural network -- in our case, it
        is the noisy image.
    Returns:
        the output of the neural network -- for us, it represents the denoised image.
    """
    # Sets batch size to one, if it is only a single image.
    if len(inputs.get_shape()) < 4:
        inputs = tf.expand_dims(inputs, 0)
    with tf.variable_scope("generator"):
        current = inputs
        # Standard Convolutional Layers:
        for _ in range(3):
            current = tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                kernel_initializer="Orthogonal",
                padding="same",
            )(current)
            current = tf.keras.layers.BatchNormalization(
                axis=3, momentum=0.0, epsilon=0.0001
            )(current)
            current = tf.nn.leaky_relu(current)
        # Residual Layers:
        for _ in range(3):
            inputs = current
            for _ in range(2):
                current = tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=(3, 3),
                    kernel_initializer="Orthogonal",
                    padding="same",
                )(current)
                current = tf.keras.layers.BatchNormalization(
                    axis=3, momentum=0.0, epsilon=0.0001
                )(current)
                current = tf.nn.leaky_relu(current)
            current = tf.identity(inputs) + current
    # U-Net based structure for deconvolution.
    current = tf.keras.layers.Conv2D(
        filters=3, kernel_size=[3, 3], kernel_initializer="Orthogonal", padding="same"
    )(current)
    current = u_net_model_fn(current)
    return current


def get_dis_conv(layers, name, filters, strides, kernel):
    """ Creates a new convolutional layer with the given name, if a layer of the same
    name does not already exists.

    Args:
        layers: a dictionary of all the layers created, with the key being their names.
        name: the name of the layer which we are trying to retrieve.
    Returns:
        a convolutional layer with the given name.
    """
    if not name in layers:
        padding = "same"
        # Using "valid" padding in the outermost layer reduces the dimension to 1.
        if name == "output_layer_conv":
            padding = "valid"
        temp = tf.keras.layers.Conv2D(
            filters=filters,
            strides=strides,
            kernel_size=kernel,
            kernel_initializer="Orthogonal",
            padding=padding,
        )
        layers[name] = temp
    return layers[name]


def get_dis_bn(layers, name):
    """ Creates a new batch normalization layer with the given name, if a layer of the
    same name does not already exists.

    Args:
        layers: a dictionary of all the layers created, with the key being their names.
        name: the name of the layer which we are trying to retrieve.
    Returns:
        a batch normalization layer with the given name.
    """
    if not name in layers:
        temp = tf.keras.layers.BatchNormalization(axis=3, momentum=0.0, epsilon=0.0001)
        layers[name] = temp
    return layers[name]


def dis_cnn_model_fn(inputs, layers):
    """ A deep convolutional neural network which acts as a binary classifier. Given a
    batch of images of size 64 x 64, it will output a number between 0 and 1, for each
    image in the batch, representing how confident it is that the image is a ground
    image (rather than one that is generated).

    Inspired by:
    https://github.com/manumathewthomas/ImageDenoisingGAN

    Args:
        input: a batch of image (either ground or generated) of size 64 x 64.
    Returns:
        a 1-D vector whose size is the batch size composed strictly of numbers between
        0 and 1.
    """
    with tf.variable_scope("discriminator"):
        # First Outer Convolutional Layer:
        current = get_dis_conv(layers, "input_layer_conv", 48, 2, [4, 4])(inputs)
        current = get_dis_bn(layers, "input_layer_bn")(current)
        current = tf.nn.relu(current)
        # First Inner Layer:
        current = get_dis_conv(layers, "conv_1", 96, 2, [4, 4])(current)
        current = get_dis_bn(layers, "bn_1")(current)
        current = tf.nn.relu(current)
        # Second Inner Layer:
        current = get_dis_conv(layers, "conv_2", 192, 2, [4, 4])(current)
        current = get_dis_bn(layers, "bn_2")(current)
        current = tf.nn.relu(current)
        # Third Inner Layer:
        current = get_dis_conv(layers, "conv_3", 384, 2, [4, 4])(current)
        current = get_dis_bn(layers, "bn_3")(current)
        current = tf.nn.relu(current)
        # Second Outer Convolutional Layer:
        current = get_dis_conv(layers, "output_layer_conv", 1, 1, [4, 4])(current)
        current = get_dis_bn(layers, "output_layer_bn")(current)
    return tf.squeeze(tf.nn.sigmoid(current))


############## IMCOMPLETE UNET MODEL INSPIRED BY: https://www.jianshu.com/p/c01136249540

def conv_relu_layer(net, numfilters, name):
    network = tf.layers.conv2d(
        net,
        activation=tf.nn.relu,
        filters=numfilters,
        kernel_size=(3, 3),
        padding="same",
        name="{}_conv_relu".format(name),
    )
    return network


def maxpool(net, name):
    network = tf.layers.max_pooling2d(
        net,
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        name="{}_maxpool".format(name),
    )
    return network


def up_conv(net, numfilters, name):
    network = tf.layers.conv2d_transpose(
        net,
        filters=numfilters,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        activation=tf.nn.relu,
        name="{}_up_conv".format(name),
    )
    return network


def copy_crop(skip_connect, net, dimension):
    print(skip_connect.shape)
    net_shape = net.get_shape()
    size = [-1, net_shape[1].value, net_shape[2].value, -1]
    if dimension:
        size = [-1, size[0], size[1], -1]
    skip_connect_crop = tf.slice(skip_connect, [0, 0, 0, 0], size)
    concat = tf.concat([skip_connect_crop, net], axis=3)
    return concat


def conv1x1(net, numfilters, name):
    return tf.layers.conv2d(
        net,
        filters=numfilters,
        strides=(1, 1),
        kernel_size=(1, 1),
        name="{}_conv1x1".format(name),
        padding="SAME",
    )


def u_net_model_fn(inputs, input_size=[64, 64]):
    """
    Taken From:
    https://www.jianshu.com/p/c01136249540
    """
    # define downsample path
    network = conv_relu_layer(inputs, numfilters=64, name="lev1_layer1")
    skip_con1 = conv_relu_layer(network, numfilters=64, name="lev1_layer2")
    network = maxpool(skip_con1, "lev2_layer1")
    network = conv_relu_layer(network, 128, "lev2_layer2")
    skip_con2 = conv_relu_layer(network, 128, "lev2_layer3")
    network = maxpool(skip_con2, "lev3_layer1")
    network = conv_relu_layer(network, 256, "lev3_layer1")
    skip_con3 = conv_relu_layer(network, 256, "lev3_layer2")
    network = maxpool(skip_con3, "lev4_layer1")
    network = conv_relu_layer(network, 512, "lev4_layer2")
    skip_con4 = conv_relu_layer(network, 512, "lev4_layer3")
    network = maxpool(skip_con4, "lev5_layer1")
    network = conv_relu_layer(network, 1024, "lev5_layer2")
    network = conv_relu_layer(network, 1024, "lev5_layer3")

    # define upsample path
    network = up_conv(network, 512, "lev6_layer1")

    network = copy_crop(skip_con4, network, input_size)

    network = conv_relu_layer(network, numfilters=512, name="lev6_layer2")

    network = conv_relu_layer(network, numfilters=512, name="lev6_layer3")

    network = up_conv(network, 256, name="lev7_layer1")
    network = copy_crop(skip_con3, network)
    network = conv_relu_layer(network, 256, name="lev7_layer2")
    network = conv_relu_layer(network, 256, "lev7_layer3")

    network = up_conv(network, 128, name="lev8_layer1")
    network = copy_crop(skip_con2, network)
    network = conv_relu_layer(network, 128, name="lev8_layer2")
    network = conv_relu_layer(network, 128, "lev8_layer3")

    network = up_conv(network, 64, name="lev9_layer1")
    network = copy_crop(skip_con1, network)
    network = conv_relu_layer(network, 64, name="lev9_layer2")
    network = conv_relu_layer(network, 64, name="lev9_layer3")
    network = conv1x1(network, 2, name="lev9_layer4")
    return network
