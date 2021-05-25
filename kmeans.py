from dataclasses import dataclass
import numpy as np
from utils import load_images, load_from_numpy


@dataclass
class Kmeans:
    X: np.array  # chaque ligne est une image
    # Y: int
    K: int
    epochs: int
    centroids = []

    def fit(self):

        updated = True
        centroids = self.select_random_k()
        centroids_label = np.empty(np.shape(self.X)[0], int)
        ep = 0

        while updated:
            print(f"{ep} / {self.epochs}")
            old_centroids = np.copy(centroids)
            centroids_label = self.center_to_cluster(centroids, centroids_label)
            centroids = self.cluster_to_center(centroids_label)
            if self.should_stop(centroids, old_centroids) or ep == self.epochs:
                updated = False
            ep += 1
        
        self.centroids = centroids

    def encode(self, point):
        distances = np.empty(len(self.centroids))
        for j, c in enumerate(self.centroids):
            distances[j] = np.linalg.norm(point - c)
        index_centroid = np.argmin(distances)
        return index_centroid



    def decode(self):
        pass

    def cluster_to_center(self, centroids_one_hot):
        centroids = np.empty([self.K, np.shape(self.X)[1]])
        for i, k in enumerate(range(self.K)):
            group_by_k = np.where(centroids_one_hot == k)
            centroids[i] = np.mean(self.X[group_by_k], axis=0)
        return centroids

    def center_to_cluster(self, centroids, centroids_label):
        for i, x in enumerate(self.X):
            distances = np.empty(len(centroids))
            for j, c in enumerate(centroids):
                distances[j] = np.linalg.norm(x - c)
            index_centroid = np.argmin(distances)
            centroids_label[i] = index_centroid
        return centroids_label

    def select_random_k(self):

        centroids = np.empty([self.K, np.shape(self.X)[1]])
        idx = np.random.choice(np.shape(self.X)[0], self.K, replace=False)
        for j, i in enumerate(idx):
            centroids[j] = self.X[i]
        return centroids

    def should_stop(self, centroids, old_centroids):
        return (centroids == old_centroids).all()


if __name__ == "__main__":
    # X, Y = load_images("./simpsons_dataset", 100, 100, True)
    X, Y = load_from_numpy()
    K = 2
    epochs = 5
    model = Kmeans(X, K, epochs)
    model.fit()
    print(model.centroids)
    model.encode(X[0])
