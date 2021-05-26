import numpy as np
from dataclasses import dataclass
from utils import load_images, load_from_numpy
import matplotlib.pyplot as plt



@dataclass
class Pca:

    X: np.ndarray
    n_components: int
    X_PCA = None
    main_composants = None

    def fit(self):
        # Center Data
        X_centered = self.X - np.mean(self.X, axis=0)

        # Covariance
        cov = np.cov(X_centered.T)
        
        # Get EigenValues And Eigen Vectors
        eigen_val, eigen_vec = np.linalg.eigh(cov)

        # Align vector and values
        sorted_index = np.argsort(eigen_val)[::-1]
        eigen_val = eigen_val[sorted_index]
        eigen_vec = eigen_vec[:,sorted_index]

        # Get the first X most important composants
        main_composants = eigen_vec[:,0:self.n_components]

        # Reduce data using main componants
        X_PCA = np.dot(main_composants.transpose(), X_centered.transpose()).transpose()
        self.X_PCA = X_PCA
        self.main_composants = main_composants
        return  X_PCA

    def custom_cov(X):
        X -= X.mean(axis=0) 
        nb_item = np.shape(X)[0] - 1 
        cov = np.dot(X.T, X) / nb_item

    def encode(self, X):
        return np.dot(X, self.main_composants)

    def decode(self, X):
        return np.dot(X, np.transpose(self.main_composants))


if __name__ == "__main__":
    X, Y = load_from_numpy()
    model = Pca(X=X, n_components=2)
    res = model.fit()

    