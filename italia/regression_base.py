import numpy as np

from .gp import GaussianProcess



class ActiveRegressionBase(object):
    """ Base class for implementations of methods for active regression.
    
    In the most basic case, a derived class only needs to implement the fetch_unlabelled() method.
    """
    
    def __init__(self, data = None, train_init = [], y_init = [], length_scale = 0.1, var = 1.0, noise = 1e-6):
        """
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - train_init: list of indices of initial training samples in data.
        
        - y_init: initial training values for the samples in train_init.
        
        - length_scale: the `sigma` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - var: the `var` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - noise: the `sigma_noise` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        """
        
        self.length_scale = length_scale
        self.var = var
        self.noise = noise
        self.fit(data, train_init, y_init)
        
        self.data = data
        self.train_init = train_init
        self.y_init = y_init
        self.gp = GaussianProcess(self.data, length_scale, var, noise)
        
        self.reset()
    
    
    def fit(self, data, train_init = [], y_init = []):
        
        self.data = data
        self.train_init = train_init
        self.y_init = y_init
        if self.data is not None:
            self.gp = GaussianProcess(
                self.data,
                self.length_scale, self.var, self.noise
            )
            self.reset()
        else:
            self.gp = None
    
    
    def reset(self):
        """ Resets the learner to its initial state directly after __init__(). """
        
        
        self.rounds = 0
        self.labeled_ids = set()
        
        if len(self.train_init) > 0:
            self.gp.fit(self.train_init, self.y_init)
            self.mean = self.gp.predict_stored()
        else:
            self.gp.reset()
            self.mean = None
    
    
    def fetch_unlabelled(self, k):
        """ Fetches a batch of unlabelled samples to be annotated from the data matrix passed to __init__().
        
        This method must be overriden in a derived class.
        
        # Arguments:
        - k: number of unlabelled samples to be chosen from the dataset.
        
        # Returns:
            list of indices of selected samples in the data matrix passed to __init__().
        """
        
        raise NotImplementedError('fetch_unlabelled() has to be implemented in a derived class.')
    
    
    def update(self, feedback):
        """ Updates the model with new annotations.
        
        # Arguments:
        - feedback: the feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples in the data matrix passed to __init__() to a numeric value.
                    None can be used to indicate that a sample has not been annotated.
        """
        
        
        ind, y = self.labeled_feedback(feedback)
        if len(ind):
        
            self.gp.update(ind, y)
            self.mean = self.gp.predict_stored()

            self.labeled_ids.update(ind)
            self.rounds += 1
    
    
    def updated_prediction(self, feedback, test_ind, cov_mode = 'full'):
        """ Obtains a prediction from the relevance model after updating it with given feedback, without actually performing the update.
        
        # Arguments:
        
        - feedback: the relevance feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples in the data matrix passed to __init__() to a numeric value.
                    None can be used to indicate that a sample has not been annotated.
        
        - test_ind: list of n indices of samples in the data matrix passed to __init__() to obtain relevance
                    predictions for.
        
        - cov_mode: one of the following values:
            - None: only predict mean
            - 'diag': predict variance for each samples
            - 'full': predict a full covariance matrix for the given samples
        
        # Returns:
            - If cov_mode is None: a length-n vector of predictive means.
            - If cov_mode is 'diag': a tuple of two length-n vectors with predictive means and variances.
            - If cov_mode is 'full': a tuple of a length-n vector with predictive means and a n-by-n covariance matrix.
        """
        
        ind, val = self.labeled_feedback(feedback)
        if len(ind) == 0:
            return self.predict_stored(test_ind, cov_mode = cov_mode)
        else:
            ind.sort()
            return self.gp.updated_prediction(ind, val, test_ind, cov_mode = cov_mode)
    
    
    def labeled_feedback(self, feedback):
        """ Partitions a dictionary with user feedback to lists of sample indices and assigned values.
        
        # Arguments:
        - feedback: the relevance feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples in the data matrix passed to __init__() to a numeric value.
                    None can be used to indicate that a sample has not been annotated.
        
        # Returns:
            a tuple with a list of indices of annotated samples and a list of assigned values.
        """
        
        ind, val = [], []
        for i, fb in feedback.items():
            if fb is not None:
                if i in self.labeled_ids:
                    raise RuntimeError('Cannot change feedback once given.')
                elif i not in self.labeled_ids:
                    ind.append(i)
                    val.append(fb)
        return ind, val