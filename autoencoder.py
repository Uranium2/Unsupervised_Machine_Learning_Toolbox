import numpy as np

from utils import load_images, load_mnist, load_from_numpy, load_mnist_PT
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class AutoEncoder:
    def __init__(self, n, m, X: np.ndarray):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X
        # self.encode = nn.Sequential(
        #     nn.Flatten(),
        #     [nn.Linear(m / i, (m / i) * 2 ) for i  in range(1, n)]
        # )
        self.shape = next(iter(X))[0].size()[1]
        self.encode_ = nn.Sequential(
            nn.Linear(self.shape, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, 4),
            nn.Sigmoid(),
            nn.Linear(4, 2)
        )

        self.decode_ = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 8),
            nn.Sigmoid(),
            nn.Linear(8, 16),
            nn.Sigmoid(),
            nn.Linear(16, 32),
            nn.Sigmoid(),
            nn.Linear(32, self.shape)
        )
        # self.decode = nn.Sequential(
        #     [nn.Linear(m / i, (m / i) * 2 ) for i  in range(n, 1, -1)]
        # )
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


# if __name__ == "__main__":
#     train_loader = load_mnist_PT()

#     model = AutoEncoder(4, 32, train_loader)
#     model.fit(2, 0.01)

