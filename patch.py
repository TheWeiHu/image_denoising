"""
Some Description Here
"""

import os
from PIL import Image
import image_slicer


def patch(input_path, save_path, dimension):
    """
    Description
    """
    for filename in os.listdir(input_path):
        if not filename.startswith("."):
            image_path = input_path + filename

            width, height = Image.open(image_path).size
            assert width == height
            assert width % dimension == 0
            n_tiles = (width / dimension) ** 2

            tiles = image_slicer.slice(image_path, n_tiles, save=False)
            image_slicer.save_tiles(
                tiles, directory=save_path, prefix=filename[0:2], format="png"
            )


if __name__ == "__main__":
    patch("./dataset/ground/", "./dataset/patch/", 64)
