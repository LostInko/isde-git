import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):

        """Estimate the centroid for each class from the training data"""
        self._class_labels = np.unique(ytr)
        self._centroids = np.zeros(shape=(self._class_labels.size, xtr.shape[1]))

        for i, label in enumerate(self._class_labels):
            self._centroids[i, :] = xtr[ytr == label, :].mean(axis=0)  # centr. for class i

        return self._centroids, self._class_labels


    def predict(self, xts):
        if self._centroids is None:
            raise ValueError("The classifier is not trained. Call fit!")
        dist_euclidean = euclidean_distances(xts, self._centroids)
        idx_min = np.argmin(dist_euclidean, axis=1)
        yc = self._class_labels[idx_min]
        return yc


