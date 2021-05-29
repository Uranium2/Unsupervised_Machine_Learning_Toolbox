import numpy as np

from utils import load_images, load_mnist, load_from_numpy, load_mnist_PT
import matplotlib.pyplot as plt
from autoencoder import AutoEncoder
import torch
import torch.nn as nn
import tqdm


class AutoEncoderConv(AutoEncoder):
    def __init__(self, layers, X: np.ndarray, activation, last_activation_encode=None, last_activation_decode=None):
        torch.manual_seed(42)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X

        self.shape = next(iter(X))[0].size()[1]
        self.init_encode_model(layers, activation, last_activation_encode)
        self.init_decode_model(layers, activation, last_activation_decode)
        self.model_ = nn.Sequential(self.encode_, self.decode_).to(self.dev)
    
    def init_encode_model(self, layers, activation, last_activation):
        encode_layers = []
        encode_layers.append(nn.Conv2d(1, layers[0], 3, stride=2, padding=1))
        encode_layers.append(activation)
        for i, l in enumerate(layers):
            if i == len(layers) - 2:
                break
            encode_layers.append(nn.Conv2d(l, layers[i + 1], 3, stride=2, padding=1))
            encode_layers.append(activation)
        encode_layers.append(nn.Conv2d(l, layers[i + 1], 3, stride=2, padding=1))
        if last_activation:
            encode_layers.append(self.last_activation_encode)
        self.encode_ = nn.Sequential(*encode_layers)

    def init_decode_model(self, layers, activation, last_activation):
        decode_layers = []
        reverse_layer = layers[::-1]
        for i, l in enumerate(reverse_layer):
            if i == len(reverse_layer) - 1:
                break

            decode_layers.append(nn.ConvTranspose2d(l, reverse_layer[i + 1], 3, stride=2, padding=1))
            decode_layers.append(activation)
        decode_layers.append(nn.ConvTranspose2d(l, 1, 3, stride=2, padding=1))
        if last_activation:
            decode_layers.append(self.last_activation_decode)
        self.decode_ = nn.Sequential(*decode_layers)


    def fit(self, epochs, lr):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=lr, weight_decay=1e-5)
        outputs = []
        for epoch in tqdm.tqdm(range(epochs)):
            for data in self.X:
                x, _ = data
                x = x.to(self.dev)
                print(self.encode_(x))
                break
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
    layers = [32, 16, 8]
    model = AutoEncoderConv(layers, train_loader, nn.Sigmoid())
    print(model.model_)
    model.fit(1, 1e-4)
