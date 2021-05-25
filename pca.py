import numpy as np
from dataclasses import dataclass
from utils import load_images, load_from_numpy
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA


@dataclass
class Pca:

    X: np.ndarray
    n_components: int

    def fit(self):
        # pca = PCA(self.n_components) 
        # return pca.fit_transform(self.X)
        # return  pca.fit_transform(self.X)
        # print("Reduced: ", PCA.fit_transform(self.X))

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
        return (X_PCA)


    def encode(self):
        pass

    def decode(self):
        pass


if __name__ == "__main__":
    X, Y = load_from_numpy()
    model = Pca(X=X, n_components=2)
    res = model.fit()

    