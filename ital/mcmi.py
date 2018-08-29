import numpy as np
import scipy.stats

import itertools
from tqdm import tqdm, trange
from multiprocessing import Pool

from .retrieval_base import ActiveRetrievalBase



class MCMI_min(ActiveRetrievalBase):
    """ Implementation of Optimistic Active Learning.
    
    Reference:
    Yhuong Guo and Russell Greiner.
    "Optimistic Active-Learning Using Mutual Information."
    IJCAI Vol. 7, 2007, pp. 823-829.
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 subsample = None, parallelized = True):
        """
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - queries: list of query data points to initially fit the GP to. If empty, the GP will not be fitted.
        
        - length_scale: the `sigma` hyper-parameter of the kernel (see documentation of ital.gp.GaussianProcess).
        
        - var: the `var` hyper-parameter of the kernel (see documentation of ital.gp.GaussianProcess).
        
        - noise: the `sigma_noise` hyper-parameter of the kernel (see documentation of ital.gp.GaussianProcess).
        
        - subsample: if set to a positive integer, the set of candidates will be restricted to a random subsample
                     of unlabeled instance of the given size.
        
        - parallelized: if set to True, conditional entropy for a set of candidate samples will be computed
                        in parallel using multiprocessing.
        """
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.subsample = subsample
        self.parallelized = parallelized
    
    
    def fetch_unlabelled(self, k, show_progress = False):
        """ Fetches a batch of unlabelled samples to be annotated from the data matrix passed to __init__().
        
        # Arguments:
        
        - k: number of unlabelled samples to be chosen from the dataset.
        
        - show_progress: if set to True, a progress bar will be shown (requires the tqdm package).
        
        # Returns:
            list of indices of selected samples in the data matrix passed to __init__().
        """
        
        self.candidates = self.get_unseen()
        if self.subsample and (self.subsample < len(self.candidates)):
            self.candidates  = np.random.choice(self.candidates, self.subsample, replace = False).tolist()
        if len(self.candidates) < k:
            k = len(self.candidates)
        
        entropy = AppendedConditionalEntropy(self)
        steps = trange(k) if show_progress else range(k)
        for it in steps:
            
            if self.parallelized and (it > 0):
                with Pool(initializer = _init_pool, initargs = (entropy,)) as p:
                    ce = p.map(_parallel_ce, self.candidates)
            else:
                ce = [entropy(i) for i in self.candidates]
            
            min_ind = np.argmin(ce)
            entropy.append(self.candidates[min_ind])
            del self.candidates[min_ind]
        
        return entropy.ret



class ConditionalEntropy(object):
    """ Helper class for MCMI_min computing conditional entropy of the relevance distribution of unlabeled data given the annotation for a certain set of samples. """
    
    def __init__(self, learner, eps = 1e-12):
        """
        # Arguments:
        
        - learner: MCMI_min instance.
        
        - eps: small number added to denominators and logs.
        """
        
        self.learner = learner
        self.eps = eps
    
    
    def __call__(self, ret):
        """ Computes the sum of conditional entropies of the relevance of all unlabeled samples given the annotations for a given set of samples.
        
        # Arguments:
        
        - ret: list of indices of samples in the candidate batch.
        
        # Returns:
            float
        """
        
        ce = None
        for reli in itertools.product([False, True], repeat = len(ret)):

            mean, var = self.learner.updated_prediction({ i : 1 if r else -1 for i, r in zip(ret, reli) }, self.learner.candidates, cov_mode = 'diag')
            
            prob_irrel = scipy.stats.norm.cdf(0, mean, np.sqrt(var))
            prob_rel = 1. - prob_irrel
            cur_ce = np.sum(prob_irrel * np.log(prob_irrel + self.eps) + prob_rel * np.log(prob_rel + self.eps))
            
            if (ce is None) or (cur_ce < ce):
                ce = cur_ce
        
        return ce



class AppendedConditionalEntropy(ConditionalEntropy):
    """ Helper class for MCMI_min computing conditional entropy of the relevance distribution of unlabeled data given the annotation for a certain set of samples.
    
    In contrast to ConditionalEntropy, this class maintains a list of already selected samples and extends this list by the candidates.
    """
    
    def __init__(self, learner, ret=[]):
        """
        # Arguments:
        
        - learner: MCMI_min instance.
        
        - ret: list of indices of already selected samples.
        """
        
        ConditionalEntropy.__init__(self, learner)
        self.set_ret(ret)
    
    
    def __call__(self, i):
        """ Computes the sum of conditional entropies of the relevance of all unlabeled samples given the annotations for the set of currently selected samples extended by a given one.
        
        # Arguments:
        - i: index of the new candidate sample.
        
        # Returns:
            float
        """
        
        return ConditionalEntropy.__call__(self, self.ret + [i])
    
    
    def set_ret(self, ret):
        """ Changes the list of already selected samples.
        
        # Arguments:
        - ret: list of indices of already selected samples.
        """
        
        self.ret = [i for i in ret]
    
    
    def append(self, i):
        """ Adds a sample to the list of selected ones.
        
        # Arguments:
        - i: index of the sample to be added.
        """
        
        self.ret.append(i)



def _init_pool(ce):
    global _ce
    _ce = ce


def _parallel_ce(i):
    return _ce(i)