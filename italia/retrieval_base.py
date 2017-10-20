import numpy as np

from .gp import GaussianProcess



class ActiveRetrievalBase(object):
    """ Base class for implementations of methods for active learning for interactive information retrieval.
    
    In the most basic case, a derived class only needs to implement the fetch_unlabelled() method.
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6):
        """
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - queries: list of query data points to initially fit the GP to. If empty, the GP will not be fitted.
        
        - length_scale: the `sigma` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - var: the `var` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - noise: the `sigma_noise` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        """
        
        self.length_scale = length_scale
        self.var = var
        self.noise = noise
        self.fit(data, queries)
    
    
    def fit(self, data, queries = []):
        
        self.data = data
        self.queries = queries
        if self.data is not None:
            self.gp = GaussianProcess(
                np.concatenate((self.data, self.queries)) if len(self.queries) > 0 else self.data,
                self.length_scale, self.var, self.noise
            )
            self.reset()
        else:
            self.gp = None
    
    
    def reset(self):
        """ Resets the learner to its initial state directly after __init__(). """
        
        
        self.rounds = 0
        self.relevant_ids = set()
        self.irrelevant_ids = set()
        
        if len(self.queries) > 0:
            self.gp.fit(np.arange(len(self.data), len(self.data) + len(self.queries)), [1] * len(self.queries))
            self.rel_mean = self.gp.predict_stored()[:len(self.data)]
        else:
            self.gp.reset()
            self.rel_mean = None
    
    
    def top_results(self, k = None):
        """ Returns the top k retrieval results for the queries passed to __init__() and the current relevance model.
        
        # Arguments:
        - k: number of samples to be retrieved from the dataset.
        
        # Returns:
            sorted list of indices of retrieved samples in the data matrix passed to __init__().
        """
        
        ind = np.argsort(self.rel_mean)[::-1]
        return ind[:k] if k is not None else ind
    
    
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
        """ Updates the relevance model with new relevance annotations.
        
        # Arguments:
        - feedback: the relevance feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples in the data matrix passed to __init__() to a numeric value.
                    Values less than 0 indicate irrelevance, values greater than 0 indicate relevance, and
                    0 itself can be used to indicate that the user has not annotated the sample.
        """
        
        
        rel, irr = self.partition_feedback(feedback)
        if len(rel) + len(irr) > 0:
        
            self.gp.update(rel + irr, np.concatenate((np.ones(len(rel)), -1 * np.ones(len(irr)))))
            self.rel_mean = self.gp.predict_stored()[:len(self.data)]

            self.relevant_ids.update(rel)
            self.irrelevant_ids.update(irr)
            self.rounds += 1
    
    
    def updated_prediction(self, feedback, test_ind, cov_mode = 'full'):
        """ Obtains a prediction from the relevance model after updating it with given feedback, without actually performing the update.
        
        # Arguments:
        
        - feedback: the relevance feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples in the data matrix passed to __init__() to a numeric value.
                    Values less than 0 indicate irrelevance, values greater than 0 indicate relevance, and
                    0 itself can be used to indicate that the user has not annotated the sample.
        
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
        
        rel, irr = self.partition_feedback(feedback)
        if len(rel) + len(irr) == 0:
            return self.predict_stored(test_ind, cov_mode=cov_mode)
        else:
            rel.sort()
            irr.sort()
            return self.gp.updated_prediction(
                rel + irr,
                np.concatenate((np.ones(len(rel)), -1 * np.ones(len(irr)))),
                test_ind,
                cov_mode = cov_mode
            )
    
    
    def partition_feedback(self, feedback):
        """ Partitions a dictionary with user feedback to lists of relevant and irrelevant samples.
        
        # Arguments:
        - feedback: the relevance feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples in the data matrix passed to __init__() to a numeric value.
                    Values less than 0 indicate irrelevance, values greater than 0 indicate relevance, and
                    0 itself can be used to indicate that the user has not annotated the sample.
        
        # Returns:
            a tuple with a list of relevant indices and a list of irrelevant indices.
        """
        
        rel, irr = [], []
        for i, fb in feedback.items():
            if fb > 0:
                if i in self.irrelevant_ids:
                    raise RuntimeError('Cannot change feedback once given.')
                elif i not in self.relevant_ids:
                    rel.append(i)
            elif fb < 0:
                if i in self.relevant_ids:
                    raise RuntimeError('Cannot change feedback once given.')
                elif i not in self.irrelevant_ids:
                    irr.append(i)
        return rel, irr