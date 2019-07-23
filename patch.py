import numpy as pn
import tensorflow as tf
import os
from PIL import Image
import image_slicer


def paASDASDtch(path, patch_size):
    for filename in os.listdir(path):
        image_path = path + filename
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=1, dct_method="INTEGER_ACCURATE")
        image = tf.expand_dims(
                image,
                axis=3
            )
        image_patches = tf.image.extract_image_patches(
            image,
            ksizes=patch_size,
            strides=patch_size,
            rates=patch_size,
            padding = "SAME"
        )

        for ind, p in enumerate(image_patches):
            tf.write_file(
                "./dataset/patches/" + filename + str(ind) + ".png", p
            )


#with tf.Session() as sess:
#    sess.run(patch("./dataset/ground", [1, 64, 64, 1]))

def crop(path, input, height, width, k, page, area):
    "https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python"

    im = Image.open(input)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                o = a.crop(area)
                o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
            except:
                pass
            k +=1
    return k


PATH = "./dataset/ground/"
SAVE_PATH = "./dataset/patches/"
k = 0
def patch(dimension):
    for filename in os.listdir(PATH):
        w, h  = Image.open(PATH + filename).size
        assert w == h

        assert w % dimension == 0
        print(filename)
        print(h)
        print(w)
        print(dimension)
        n_tiles = (w / dimension) **2


        image_path = PATH + filename
        tiles = image_slicer.slice(image_path, n_tiles, save=False)
        image_slicer.save_tiles(tiles, directory=SAVE_PATH, prefix=filename, format='png')


patch(64)