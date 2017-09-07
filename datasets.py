import numpy as np
import csv

import sklearn.datasets
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mvn



def load_dataset(dataset_name, **kwargs):
    
    class_name = dataset_name + 'Dataset'
    if class_name not in globals():
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
    return globals()[class_name](**kwargs)



class Dataset(object):
    
    def __init__(self, X, y, test_size = 0.2):
        
        self.X = np.array(X)
        self.y = np.array(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = 0, stratify = self.y)
        
        self.imgs = self.imgs_train = self.imgs_test = None
        
        self._preprocess()
    
    
    def _preprocess(self):
        
        self.X_max, self.X_min = self.X_train.max(), self.X_train.min()
        self.X_train_norm = (self.X_train - self.X_min) / (self.X_max - self.X_min)
        self.X_test_norm = (self.X_test - self.X_min) / (self.X_max - self.X_min)
        
        self.labels = np.unique(self.y)
        self.class_relevance = { lbl : (2 * (self.y_train == lbl) - 1, 2 * (self.y_test == lbl) - 1) for lbl in self.labels }



class ToyDataset(Dataset):
    
    def __init__(self, size_factor = 10, test_size = 0.5, **kwargs):
        
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
    
    def __init__(self, **kwargs):
        
        X, y = sklearn.datasets.load_iris(return_X_y = True)
        Dataset.__init__(self, X, y, **kwargs)



class WineDataset(Dataset):
    
    def __init__(self, data_file, **kwargs):
        
        X = np.loadtxt(data_file, delimiter = ',', dtype = float)[:,1:]
        y = np.loadtxt(data_file, delimiter = ',', dtype = int, usecols = 0)
        Dataset.__init__(self, X, y, **kwargs)



class LeafDataset(Dataset):
    
    def __init__(self, data_file, test_size = 0.5, **kwargs):
        
        X = np.loadtxt(data_file, delimiter = ',', dtype = float)[:,2:]
        y = np.loadtxt(data_file, delimiter = ',', dtype = int, usecols = 0)
        Dataset.__init__(self, X, y, test_size=test_size, **kwargs)



class ButterflyDataset(Dataset):
    
    def __init__(self, data_file, **kwargs):
        
        data = np.load(data_file)
        self.X_train, self.y_train = data['X_train'], data['y_train']
        self.X_test, self.y_test = data['X_test'], data['y_test']
        
        self.X = np.concatenate((self.X_train, self.X_test))
        self.y = np.concatenate((self.y_train, self.y_test))
        
        self.imgs = self.imgs_train = self.imgs_test = None
        
        self._preprocess()



class USPSDataset(Dataset):
    
    def __init__(self, train_data_file, test_data_file, **kwargs):
        
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