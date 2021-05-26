import numpy as np

from utils import load_images, load_mnist, load_from_numpy, load_mnist_PT
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class AutoEncoder:
    def __init__(self, layers, X: np.ndarray):
        torch.manual_seed(42)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X

        self.shape = next(iter(X))[0].size()[1]

        encode_layers = []

        encode_layers.append(nn.Linear(self.shape, layers[0]))
        for i, l in enumerate(layers):
            if i == len(layers) - 1:
                break
            encode_layers.append(nn.Linear(l, layers[i + 1]))


        self.encode_ = nn.Sequential(*encode_layers)

        decode_layers = []
        reverse_layer = layers[::-1]
        for i, l in enumerate(reverse_layer):
            if i == len(reverse_layer) - 1:
                break
            decode_layers.append(nn.Linear(l, reverse_layer[i + 1]))
        decode_layers.append(nn.Linear(reverse_layer[-1], self.shape))

        self.decode_ = nn.Sequential(*decode_layers)

        self.model_ = nn.Sequential(self.encode_, self.decode_)

    def fit(self, epochs, lr):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=lr, weight_decay=1e-5
        )
        outputs = []
        for epoch in range(epochs):
            print(epoch)
            for x, _ in self.X:
                out = self.model_(x)
                loss = criterion(out, x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def encode(self, X):
        return self.encode_(X)


    def decode(self, X):
        return self.decode_(X)


if __name__ == "__main__":
    train_loader = load_mnist_PT()
    layers = [32, 16, 8, 4, 2]
    model = AutoEncoder(layers, train_loader)
    # model.fit(2, 0.01)
    print(model.encode_)
    print(model.decode_)

