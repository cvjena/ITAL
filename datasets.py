import numpy as np
import csv

import sklearn.datasets
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mvn



def load_dataset(dataset_name, **kwargs):
    """ Instantiates a dataset by its name.
    
    # Arguments:
    - dataset_name: The name of the dataset. Appending 'Dataset' to it should result in the name of the class to be intantiated.
    
    Additional arguments will be passed through to the constructor of the dataset.
    
    # Returns:
        object
    """
    
    class_name = dataset_name + 'Dataset'
    if class_name not in globals():
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
    return globals()[class_name](**kwargs)



class Dataset(object):
    """ A dataset.
    
    # Properties:
    
    - X: array of all samples.
    
    - y: array with labels of all samples.
    
    - X_train: array with training data.
    
    - y_train: array with training labels.
    
    - X_test: array with validation data.
    
    - y_test: array with validation labels.
    
    - X_max: maximum value in X_train.
    
    - X_min: minimum value in X_train.
    
    - X_train_norm: X_train scaled to [0,1].
    
    - X_test_norm: X_test scaled by the same parameters used to obtain X_train_norm.
    
    - labels: list of unique labels.
    
    - class_relevance: dictionary mapping labels to arrays specifying whether a sample
                       is relevant for that label. Class relevance is given as 1 or -1.
    """
    
    def __init__(self, X, y, test_size = 0.2):
        """ Initializes a dataset from given data, automatically splitting it into training and validation set.
        
        # Arguments:
        
        - X: array of all samples.
        
        - y: array with labels of all samples.
        
        - test_size: either a float specifying the fraction of the data to be used for
                     validation or an integer specifying the absolute number of
                     validation samples.
        """
        
        self.X = np.array(X)
        self.y = np.array(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = 0, stratify = self.y)
        
        self.imgs = self.imgs_train = self.imgs_test = None
        
        self._preprocess()
    
    
    def _preprocess(self):
        """ Computes X_max, X_min, X_train_norm, X_test_norm, labels, and class_relevance. """
        
        self.X_max, self.X_min = self.X_train.max(), self.X_train.min()
        self.X_train_norm = (self.X_train - self.X_min) / (self.X_max - self.X_min)
        self.X_test_norm = (self.X_test - self.X_min) / (self.X_max - self.X_min)
        
        self.labels = np.unique(self.y)
        self.class_relevance = { lbl : (2 * (self.y_train == lbl) - 1, 2 * (self.y_test == lbl) - 1) for lbl in self.labels }



class StoredDataset(Dataset):
    """ Loads a dataset from a numpy file. """
    
    def __init__(self, data_file, **kwargs):
        """ Loads the dataset.
        
        # Arguments:
        - data_file: path to a .npz file containing X_train, y_train, X_test, and y_test.
        """
        
        data = np.load(data_file)
        self.X_train, self.y_train = data['X_train'], data['y_train']
        self.X_test, self.y_test = data['X_test'], data['y_test']
        
        self.X = np.concatenate((self.X_train, self.X_test))
        self.y = np.concatenate((self.y_train, self.y_test))
        
        self.imgs = self.imgs_train = self.imgs_test = None
        
        self._preprocess()



class ToyDataset(Dataset):
    """ Generates a toy dataset. """
    
    def __init__(self, size_factor = 10, test_size = 0.5, **kwargs):
        """ Initializes the dataset.
        
        # Arguments:
        - size_factor: controls to size of the dataset, which will be 17 * size_factor.
        """
        
        np.random.seed(0)
        X = np.concatenate([
                mvn.rvs([20.0, 30.0], np.random.randn(2,2) + np.eye(2) * 2, 2 * size_factor),
                mvn.rvs([10.0, 10.0], np.random.randn(2,2) + np.eye(2) * 2, 3 * size_factor),
                mvn.rvs([30.0, 15.0], np.random.randn(2,2) + np.eye(2) * 2, 6 * size_factor),
                mvn.rvs([15.0, 20.0], np.random.randn(2,2) + np.eye(2) * 2, 4 * size_factor),
                mvn.rvs([25.0, 30.0], np.random.randn(2,2) + np.eye(2) * 2, 2 * size_factor)
            ])
        y = np.concatenate([np.ones(5 * size_factor, dtype=int), np.zeros(12 * size_factor, dtype=int)])
        
        Dataset.__init__(self, X, y, test_size=test_size, **kwargs)



class IrisDataset(Dataset):
    """ Interface to the Iris dataset. """
    
    def __init__(self, **kwargs):
        
        X, y = sklearn.datasets.load_iris(return_X_y = True)
        Dataset.__init__(self, X, y, **kwargs)



class WineDataset(Dataset):
    """ Interface to the UCI Wine dataset.
    
    https://archive.ics.uci.edu/ml/datasets/wine
    """
    
    def __init__(self, data_file, **kwargs):
        """ Loads the Wine dataset.
        
        # Arguments:
        - data_file: path to wine.data.
        """
        
        X = np.loadtxt(data_file, delimiter = ',', dtype = float)[:,1:]
        y = np.loadtxt(data_file, delimiter = ',', dtype = int, usecols = 0)
        Dataset.__init__(self, X, y, **kwargs)



class LeafDataset(Dataset):
    """ Interface to the UCI Leaf dataset.
    
    https://archive.ics.uci.edu/ml/datasets/leaf
    """
    
    def __init__(self, data_file, test_size = 0.5, **kwargs):
        """ Loads the Wine dataset.
        
        # Arguments:
        - data_file: path to leaf.csv.
        """
        
        X = np.loadtxt(data_file, delimiter = ',', dtype = float)[:,2:]
        y = np.loadtxt(data_file, delimiter = ',', dtype = int, usecols = 0)
        Dataset.__init__(self, X, y, test_size=test_size, **kwargs)



class USPSDataset(Dataset):
    """ Interface to the USPS dataset.
    
    https://www-i6.informatik.rwth-aachen.de/~keysers/usps.html
    """
    
    def __init__(self, train_data_file, test_data_file, **kwargs):
        """ Loads the USPS dataset.
        
        # Arguments:
        - train_data_file: path to usps_train.jf.
        - test_data_file: path to usps_test.jf.
        """
        
        self.X_train, self.y_train = self._read_usps(train_data_file)
        self.X_test, self.y_test = self._read_usps(test_data_file)
        
        self.X = np.concatenate((self.X_train, self.X_test))
        self.y = np.concatenate((self.y_train, self.y_test))
        
        self.imgs = self.X.reshape(-1, 16, 16)
        self.imgs_train = self.X_train.reshape(-1, 16, 16)
        self.imgs_test = self.X_test.reshape(-1, 16, 16)
        
        self._preprocess()
    
    
    def _read_usps(self, data_file):
        
        with open(data_file) as f:
            num_classes, num_features = [int(x) for x in f.readline().strip().split()]
            X, y = [], []
            for line in f:
                if (line.strip() == '') or (line.strip() == '-1'):
                    break
                data = line.strip().split()
                y.append(int(data[0]))
                X.append([float(x) for x in data[1:]])
        
        return np.array(X), np.array(y)