import numpy as np
from scipy.spatial.distance import cdist


class KNearestNeighbor():
    """ a kNN classifier with L1/L2 distance """
    
    def __init__(self):
        pass
    
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
             consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X, k=1, distance_type='L2'):
        """
        Predict labels for test data using this classifier.
        
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The mumber of nearest neighbors that vote for the predicted labels.
        - distance_type: The type of distance between the samples. Implement
                         for L1 and L2.
        
        Returns:
        - y: A numpy array of shape (num_test,) containing predited labels for the
             test data, where y[i] is the predicted label for the test point X[i]
        """
        if distance_type == 'L1':
            dists = self.compute_distances_L1(X)
        elif distance_type == 'L2':
            dists = self.compute_distances_L2(X)
        else:
            raise ValueError('Invalid value {} for distance_type'.format(distance_type))
            
        return self.predict_labels(dists, k=k)
    
    def compute_distances_L2(self, X):
        """
        Compute the L2 distance between each test point in X and each training point
        in self.X_train
        
        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.
        
        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
                 is the Euclidean distance between the ith test point and the jth
                 training point.
        """
        dists = np.sqrt(np.sum(X ** 2, axis=1)[:, np.newaxis] - 2 * X.dot(self.X_train.T) +
                np.sum(self.X_train ** 2, axis=1))
        return dists
    
    def compute_distances_L1(self, X):
        """
        Compute the L2 distance between each test point in X and each training point
        in self.X_train
        
        Input / Output: Same as compute_distance_L2
        """
        dists = cdist(X, self.X_train, 'cityblock')
        
        return dists
    
    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        
        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
                 gives the distance between the ith test point and the jth
                 training point.
                 
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
             test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            indices = np.argsort(dists[i, :])[:k]
            # A list of length k storing the labels of the k nearest neighbors
            # to the ith test point.
            closest_y = self.y_train[indices]
            labels, counts = np.unique(closest_y, return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
        
        return y_pred
            