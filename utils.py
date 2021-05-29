import os
import glob
from PIL import Image
import numpy as np
from tensorflow import keras
import torch

from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import tensorflow as tf
import matplotlib.pyplot as plt


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


def load_from_numpy(path_X, path_Y):
    return np.load(path_X), np.load(path_Y)


def load_mnist_PT():
    mnist_data = datasets.MNIST(
        ".",
        True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
        ),
    )
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)
    return train_loader


def load_mnist_CNN_PT(batch_size):
    mnist_data = datasets.MNIST(
        ".",
        True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        mnist_data, batch_size=batch_size, shuffle=True
    )
    return train_loader


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = np.reshape(
        x_train, (np.shape(x_train)[0], np.shape(x_train)[1] * np.shape(x_train)[2])
    )
    x_train = x_train / 255.0
    return x_train, y_train


def load_mnist2():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    return tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(256)


class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_path, image_width, image_height):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms

        Args:
            folder_path (string): path to image folder
        """
        self.image_height = image_height
        self.image_width = image_width
        # Get image list
        self.image_list = glob.glob(folder_path + "/*/*.jpg")
        self.image_list_label = np.array(
            ["\\".join(img.split("\\")[-2:-1]) for img in self.image_list]
        )
        # Calculate len
        self.data_len = len(self.image_list)

        self.X = []

        for index in range(self.data_len):
            single_image_path = self.image_list[index]
            # Open image
            im_as_im = Image.open(single_image_path)
            
            im_as_im = np.array(
                im_as_im.resize((self.image_width, self.image_height), Image.ANTIALIAS)
            ).flatten()

            im_as_np = np.asarray(im_as_im) / 255

            im_as_np = np.expand_dims(im_as_np, 0)

            # Transform image to tensor, change data type
            self.X.append(torch.from_numpy(im_as_np).float())


    def __getitem__(self, index):
        # Get image name from the pandas df

        x = self.X[index]
        label = self.image_list_label[index]
        return (x, label)

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    X, Y = load_images("./simpsons_dataset", 28, 28, True)
    # X, Y = load_from_numpy("X.npy", "Y.npy")
    # print(np.shape(X))
    # print(np.shape(X[0]))
    # print(np.reshape(X[0], (25, 25, 3)))
    # plt.imshow(np.reshape(X[0], (25, 25, 3)))
    # plt.show()
    # X, Y = load_mnist()
    # print(X, Y)
    # X = CustomDatasetFromFile("./simpsons_dataset", 25, 25)
    # print(X.__getitem__(10))
