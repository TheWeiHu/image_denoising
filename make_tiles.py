"""
A tool used for cutting images into grids. Given a square image and a dimension, the
image will be cut into tiles of that dimension (if it is possible -- i.e. the
dimension of the input image is a multiple of the chosen dimension).
"""

import os
from PIL import Image
import image_slicer


def make_tiles(input_path, save_path, dimension):
    """Given a square image and a dimension, the image will be cut into tiles of that
    dimension, whenever possible.

    Args:
        input_path: the file path which lead to the image which is to be cut up.
        save_path: the directory in which the cut up pieces will be stored.
        dimension: the prescribed dimension of the cut up tiles.
    """
    for filename in os.listdir(input_path):
        if not filename.startswith("."):
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


if __name__ == "__main__":
    make_tiles("./dataset/train/", "./dataset/patch/", 64)
