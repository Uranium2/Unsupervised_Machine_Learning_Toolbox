import torch
import numpy as np
from dataclasses import dataclass
from utils import load_images, load_from_numpy
import matplotlib.pyplot as plt


@dataclass
class AutoEncoder:

    encoder = None
    decoder = None

    def init_models(self, n, m):
        self.encode = nn.Sequential(
            nn.Flatten(),
            [nn.Linear(m / i ) for i  in range(1, n)]
        )

        self.decode = nn.Sequential(
            [nn.Linear(m / i ) for i  in range(n, 1, -1)]
        )
    def fit(self):
        pass

    def encode(self, X):
        pass

    def decode(self):
        pass