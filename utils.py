import os
import glob
from PIL import Image
import numpy as np
from tensorflow import keras


def load_images(path, image_width, image_height, save=False):

    images = glob.glob(path + "/*/*.jpg")
    images_y = np.array(["\\".join(img.split("\\")[-2:-1]) for img in images])
    _, _, Y = np.unique(images_y, return_index=True, return_inverse=True)

    X = np.empty((len(images), image_width * image_height * 3), int)
    for i, img in enumerate(images):
        filepath = os.path.join(path, img)
        data = Image.open(filepath)

        data = np.array(
            data.resize((image_width, image_height), Image.ANTIALIAS)
        ).flatten()

        X[i] = data

    if save:
        np.save("X", X)
        np.save("Y", Y)
    return X, Y


def load_from_numpy():
    return np.load("X.npy"), np.load("Y.npy")


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_train, y_train


if __name__ == "__main__":
    # Xload_images("./simpsons_dataset", 100, 100, True), Y =
    # X, Y = load_from_numpy()
    X, Y = load_mnist()
    print(X, Y)
