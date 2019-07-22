import os
import numpy as np
import cv2


def gaussian_noise(image_path):
    """
    https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    row, col = image.shape
    mean = 0
    var = 0.5
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def verify():
    """
    Verifies that the noise generated is indeed Gaussian.
    """
    for filename in os.listdir("./dataset/ground"):
        if not filename.startswith("."):
            for i in range(1000):
                image = cv2.imread("./dataset/ground/" + filename, cv2.IMREAD_GRAYSCALE)
                noise_img_new = cv2.imread(
                    "./dataset/noisy/noisy_" + str(i) + "_" + filename,
                    cv2.IMREAD_GRAYSCALE,
                )
                print(
                    sum(
                        int(i) - int(j)
                        for i, j in zip(
                            list(np.reshape(noise_img_new, -1)),
                            list(np.reshape(image, -1)),
                        )
                    )
                )


def main():
    for filename in os.listdir("./dataset/ground"):
        if not filename.startswith("."):
            for i in range(1000):
                noise_img = gaussian_noise("./dataset/ground/" + filename)
                cv2.imwrite(
                    "./dataset/noisy/noisy_" + str(i) + "_" + filename, noise_img
                )
    # verify()


if __name__ == "__main__":
    main()
