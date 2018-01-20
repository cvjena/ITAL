import numpy as np
import scipy.stats

from multiprocessing import Pool

from .retrieval_base import ActiveRetrievalBase
from .gp import invh



def reduced_inv(K, K_inv, i):
    """ Updates the inverse of a kernel matrix by reducing one row and column.
    
    # Arguments:
    
    - K: the full kernel matrix.
    
    - K_inv: the pre-computed inverse of `K`.
    
    - i: index of the row and column to be removed from `K`
    
    # Returns:
        the inverse of `K[np.ix_(ind, ind)]` with `ind = np.setdiff1d(np.arange(K.shape[0]), [i])`
    """
    
    t = K.shape[0]
    u = np.concatenate([[-1], np.zeros(t - 1)])[:,None]
    oids = np.concatenate([np.arange(i), np.arange(i+1, t)])
    v = np.concatenate([[K[i,i] - 1], K[i, oids]])[:,None]
    w = np.concatenate([[0], K[oids, i]])[:,None]
    
    ind = np.concatenate([[i], oids])
    invA = K_inv[np.ix_(ind, ind)]
    invA -= np.linalg.multi_dot([invA, u, v.T, invA]) / (1 + np.linalg.multi_dot([v.T, invA, u]))
    invA -= np.linalg.multi_dot([invA, w, u.T, invA]) / (1 + np.linalg.multi_dot([u.T, invA, w]))
    return invA[1:,1:]



class AdaptAL(ActiveRetrievalBase):
    """ Implementation of Adaptive Active Learning.
    
    Reference:
    Xin Li and Yhuong Guo.
    "Adaptive Active-Learning for Image Classification."
    CVPR 2013, pp. 859-866.
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 subsample = None, parallelized = True, betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        """
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - queries: list of query data points to initially fit the GP to. If empty, the GP will not be fitted.
        
        - length_scale: the `sigma` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - var: the `var` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - noise: the `sigma_noise` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - subsample: if set to a positive integer, the set of candidates will be restricted to a random subsample
                     of unlabeled instance of the given size.
        
        - parallelized: if set to True, conditional entropy for a set of candidate samples will be computed
                        in parallel using multiprocessing.
        
        - betas: list of beta values (combination coefficient) to be tested.
        """
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.subsample = subsample
        self.parallelized = parallelized
        self.betas = betas
    
    
    def fetch_unlabelled(self, k):
        """ Fetches a batch of unlabelled samples to be annotated from the data matrix passed to __init__().
        
        # Arguments:
        
        - k: number of unlabelled samples to be chosen from the dataset.
        
        # Returns:
            list of indices of selected samples in the data matrix passed to __init__().
        """
        
        
        candidates = np.array([i for i in range(len(self.data)) \
                               if (i not in self.relevant_ids) \
                               and (i not in self.irrelevant_ids)])
        if self.subsample and (self.subsample < len(candidates)):
            candidates  = np.random.choice(candidates, self.subsample, replace = False)
        if len(candidates) < k:
            k = len(candidates)
        
        # Compute entropy and information density for all samples
        rel_mean, rel_var = self.gp.predict_stored(candidates, cov_mode = 'diag')
        entropy = self.entropy(rel_mean, rel_var)
        density = self.information_density(candidates)
        
        # For every beta, select k samples maximizing a combination of entropy and density weighted with beta
        scores = np.stack([(entropy ** beta) * (density ** (1. - beta)) for beta in self.betas])
        max_ind = np.unique(np.argpartition(-scores, k - 1, axis = -1)[:,:k].ravel())
        
        # Choose the k samples from that reduced set minimizing expected classification error
        if len(max_ind) <= k:
            return candidates[max_ind].tolist()
        else:
            scores = np.array([self.expected_classification_error(candidates, rel_mean, rel_var, candidates[i], i) for i in max_ind])
            min_ind = np.argpartition(scores, k - 1)[:k]
            return candidates[max_ind[min_ind]].tolist()
    
    
    def entropy(self, mean, var):
        """ Computes the entropy of the relevance of some samples.
        
        # Arguments:
        
        - mean: vector of predictive means of the samples.
        
        - var: vector of predictive variances of the samples.
        
        # Returns:
            vector with the entropies of the samples
        """
        
        prob_irr = np.maximum(1e-8, np.minimum(1.0 - 1e-8, scipy.stats.norm.cdf(0, mean, np.sqrt(var))))
        return -1 * (prob_irr * np.log(prob_irr) + (1.0 - prob_irr) * np.log(1.0 - prob_irr))
    
    
    def information_density(self, candidates):
        """ Computes the information density of a set of samples.
        
        # Arguments:
        
        - candidates: vector with indices of the samples to compute information density for.
        
        # Returns:
            vector with the information densities of the samples
        """
        
        K = self.gp.K_all[np.ix_(candidates, candidates)] + self.gp.noise * np.eye(len(candidates))
        K_inv = invh(K)
        
        if self.parallelized:
            
            with Pool(initializer = _init_pool, initargs = (K, K_inv)) as p:
                sigma_updated = p.map(_parallel_density, range(len(candidates)))
            
        else:
            
            sigma_updated = np.ndarray((len(candidates),))
            for i in range(len(candidates)):
                k = np.concatenate([K[i,:i], K[i,i+1:]])
                K_inv_red = reduced_inv(K, K_inv, i)
                sigma_updated[i] = K[i,i] - np.dot(k, np.dot(K_inv_red, k.T))
        
        return np.log(np.diag(K) / np.maximum(1e-6, sigma_updated)) / 2
    
    
    def expected_classification_error(self, unlabeled, mean, var, ret, ret_ind):
        """ Computes the expected classification error of a given sample.
        
        # Arguments:
        
        - unlabeled: indices of all unlabeled samples.
        
        - mean: predictive mean of all unlabeled samples.
        
        - var: predictive variance of all unlabeled samples.
        
        - ret: index of the sample whose expected classification error is to be computed in the dataset.
        
        - ret_ind: index of the sample whose expected classification error is to be computed in `mean` and `var`.
        
        # Returns:
            float
        """
        
        expected_label = (mean > 0)
        prob_irr = np.maximum(1e-8, np.minimum(1.0 - 1e-8, scipy.stats.norm.cdf(0, mean[ret_ind], np.sqrt(var[ret_ind]))))
        ret_ind_compl = np.setdiff1d(np.arange(len(unlabeled)), [ret_ind])
        
        err = 0
        for fb in [True, False]:
            updated_mean, updated_var = self.gp.updated_prediction([ret], [fb], unlabeled[ret_ind_compl], cov_mode = 'diag')
            updated_prob_irr = np.maximum(1e-8, np.minimum(1.0 - 1e-8, scipy.stats.norm.cdf(0, updated_mean, np.sqrt(updated_var))))
            err += (1 - prob_irr if fb else prob_irr) * np.sum(np.where(expected_label[ret_ind_compl], updated_prob_irr, 1. - updated_prob_irr))
        return err



def _init_pool(K, K_inv):
    
    global _K
    global _K_inv
    _K = K
    _K_inv = K_inv


def _parallel_density(i):
    
    k = np.concatenate([_K[i,:i], _K[i,i+1:]])
    K_inv_red = reduced_inv(_K, _K_inv, i)
    return _K[i,i] - np.dot(k, np.dot(K_inv_red, k.T))
