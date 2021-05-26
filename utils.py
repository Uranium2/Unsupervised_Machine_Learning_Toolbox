import os
import glob
from PIL import Image
import numpy as np
from tensorflow import keras
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms



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
    x_train = np.reshape(x_train, (np.shape(x_train)[0], np.shape(x_train)[1] * np.shape(x_train)[2]))
    return x_train, y_train


class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_path):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms

        Args:
            folder_path (string): path to image folder
        """
        # Get image list
        self.image_list = glob.glob(folder_path + "/*/*.jpg")
        self.image_list_label = np.array(["\\".join(img.split("\\")[-2:-1]) for img in self.image_list])
        # Calculate len
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        im_as_im = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        im_as_np = np.asarray(im_as_im)/255
        # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension
        im_as_np = np.expand_dims(im_as_np, 0)
        # Some preprocessing operations on numpy array
        # ...
        # ...
        # ...

        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(im_as_np).float()

        label = self.image_list_label[index]
        return (im_as_ten, label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # Xload_images("./simpsons_dataset", 100, 100, True), Y =
    # X, Y = load_from_numpy()
    # X, Y = load_mnist()
    # print(X, Y)
    X = CustomDatasetFromFile("./simpsons_dataset")
    print(X.__getitem__(10))
