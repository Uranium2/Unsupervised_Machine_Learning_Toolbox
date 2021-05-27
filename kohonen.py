from utils import load_mnist
import numpy as np
import matplotlib.pyplot as plt

class Kohonen():

    def __init__(self, X, size_map):
        self.X = X
        self.size_map = size_map
        K = self.size_map[0] * self.size_map[1]
        W = self.select_random_k(K)
        self.Kmap = np.reshape(W, (self.size_map[0], self.size_map[1], self.X.shape[1]))

    def select_random_k(self, K):
        centroids = np.empty([K, np.shape(self.X)[1]])
        idx = np.random.choice(np.shape(self.X)[0], K, replace=False)
        for j, i in enumerate(idx):
            centroids[j] = self.X[i]
        return centroids

    def find_closest_k(self, example):
        distances = np.empty(self.size_map[0] * self.size_map[1])
        j = 0
        for y in range(self.Kmap.shape[0]):
            for x in range(self.Kmap.shape[1]):
                distances[j] = np.linalg.norm(self.Kmap[y, x] - example)
                j += 1
        index_centroid = np.argmin(distances)
        
        closest_x = int(index_centroid % self.size_map[1])
        closest_y = int(index_centroid / self.size_map[1])
        return closest_y, closest_x
    
    def update_kmap(self, closest_y, closest_x, example, lr, gamma):
        for y in range(self.Kmap.shape[0]):
            for x in range(self.Kmap.shape[1]):
                self.Kmap[y, x] = self.Kmap[y, x] + lr * (np.exp(- np.linalg.norm(np.array([y, x]) - np.array([closest_y, closest_x]) ) / 2 * gamma)) * (example - self.Kmap[y, x])
    
    def fit(self, epochs, lr, gamma):
        for i in range(epochs) :
            if i % 1000 == 0:
                print(i)
            idx = np.random.choice(np.shape(self.X)[0], 1, replace=False)
            example = self.X[idx]
            closest_y, closest_x = self.find_closest_k(example)
            self.update_kmap(closest_y, closest_x, example, lr, gamma)
    
    def display_grid(self):
        f, fig = plt.subplots(self.size_map[0], self.size_map[1])
        for y in range(self.Kmap.shape[0]):
            for i, x in enumerate(range(self.Kmap.shape[1])):
                fig[y, x].imshow(np.reshape(self.Kmap[y, x], (28,28)), cmap='gray')