import numpy as np
import scipy.stats

import itertools

from .retrieval_base import ActiveRetrievalBase



class RandomRetrieval(ActiveRetrievalBase):
    """ Selects samples at random. """
    
    def fetch_unlabelled(self, k):
        
        candidates = [i for i in range(len(self.data)) \
                      if (i not in self.relevant_ids) \
                      and (i not in self.irrelevant_ids)]
        
        return np.random.choice(candidates, min(k, len(candidates)), replace = False)



class BorderlineSampling(ActiveRetrievalBase):
    """ Selects samples with minimum absolute predictive mean. """
    
    def fetch_unlabelled(self, k):
        
        ranking = np.argsort(np.abs(self.rel_mean))
        ret = []
        for i in ranking:
            if (i not in self.relevant_ids) and (i not in self.irrelevant_ids):
                ret.append(i)
                if len(ret) >= k:
                    break
        return ret



class VarianceSampling(ActiveRetrievalBase):
    """ Selects samples with maximum predictive variance.
    
    If `use_correlations` is set to `True`, the covariance to other samples in the selected batch will also
    be taken into account by computing the score of a given batch of samples as the sum of their variance
    minus the sum of their covariance. Samples will be selected in a greedy fashion, starting with the one
    with the highest predictive variance and extending the batch successively.
    """
    
    def __init__(self, data, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 use_correlations = False):
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.use_correlations = use_correlations
    
    
    def fetch_unlabelled(self, k):
        
        _, rel_var = self.gp.predict_stored(cov_mode = 'diag')
        rel_var = rel_var[:len(self.data)]
        
        if self.use_correlations:
            
            ret = [max(range(rel_var.size), key = lambda i: rel_var[i] if (i not in self.relevant_ids) and (i not in self.irrelevant_ids) else 0)]
            for l in range(1, k):
                candidates = [i for i in range(rel_var.size) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids) and (i not in ret)]
                if len(candidates) == 0:
                    break
                covs = self.gp.predict_cov_batch(ret, candidates)
                ti, tj = np.tril_indices(covs.shape[1], -1)
                scores = np.diagonal(covs, 0, 1, 2).sum(axis = -1) - covs[:,ti,tj].sum(axis = -1)
                ret.append(candidates[np.argmax(scores)])
            
        else:
            
            ranking = np.argsort(rel_var)[::-1]
            ret = []
            for i in ranking:
                if (i not in self.relevant_ids) and (i not in self.irrelevant_ids):
                    ret.append(i)
                    if len(ret) >= k:
                        break
        
        return ret



class UncertaintySampling(ActiveRetrievalBase):
    """ Selects samples with minimum certainty
    
    Certainty is defined as: |mu| / sqrt(sigma^2 + sigma_noise^2)
    
    Reference:
    Ashish Kapoor, Kristen Grauman, Raquel Urtasun and Trevor Darrell.
    "Active Learning with Gaussian Processes for Object Categorization."
    International Conference on Computer Vision (ICCV), 2007.
    """
    
    def fetch_unlabelled(self, k):
        
        mean, variance = self.gp.predict_stored(cov_mode = 'diag')
        ranking = np.argsort(np.abs(mean) / np.sqrt(variance + self.gp.noise))
        ret = []
        for i in ranking:
            if (i not in self.relevant_ids) and (i not in self.irrelevant_ids):
                ret.append(i)
                if len(ret) >= k:
                    break
        return ret



class EntropySampling(ActiveRetrievalBase):
    """ Selects batches of samples with maximum entropy.
    
    Reference:
    Ksenia Konyushkova, Raphael Sznitman and Pascal Fua.
    "Geometry in Active Learning for Binary and Multi-class Image Segmentation."
    arXiv:1606.09029v2.
    
    For batch sampling, this implementation uses the joint distribution of the samples in the
    batch for computing the batch entropy.
    """
    
    def fetch_unlabelled(self, k):
        
        rel_mean, rel_var = self.gp.predict_stored(cov_mode = 'diag')
        rel_mean = rel_mean[:len(self.data)]
        rel_var = rel_var[:len(self.data),]
        
        ret = [max(range(rel_mean.size), key = lambda i: self.single_entropy(rel_mean[i], rel_var[i]) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids) else -np.inf)]
        for l in range(1, k):
            candidates = [i for i in range(rel_mean.size) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids) and (i not in ret)]
            if len(candidates) == 0:
                break
            covs = self.gp.predict_cov_batch(ret, candidates)
            ret.append(max(candidates, key = lambda i: self.batch_entropy(rel_mean[ret+[i]], covs[i])))
        
        return ret
    
    
    def single_entropy(self, mean, var):
        
        prob_irr = max(1e-8, min(1.0 - 1e-8, scipy.stats.norm.cdf(0, mean, np.sqrt(var))))
        return -1 * (prob_irr * np.log(prob_irr) + (1.0 - prob_irr) * np.log(1.0 - prob_irr))
    
    
    def batch_entropy(self, mean, cov):
        
        stdev = np.sqrt(np.diag(cov))
        pivot = -mean / stdev

        i, j = np.tril_indices(cov.shape[0], -1)
        correl = cov[i, j] / (stdev[i] * stdev[j])

        entropy = 0.0
        for rel in itertools.product([False, True], repeat = len(mean)):
            err, pr, info = scipy.stats.mvn.mvndst(pivot, pivot, np.array(rel, dtype = int), correl,
                                                   maxpts = len(rel) * 100, abseps = 1e-4, releps = 1e-4)
            if pr > 1e-12:
                entropy += pr * np.log(pr)

        return -1 * entropy



class EMOC(ActiveRetrievalBase):
    """ Selects samples with maximum expected model output change (EMOC).
    
    Reference:
    Alexander Freytag, Erik Rodner and Joachim Denzler.
    "Selecting Influential Examples: Active Learning with Expected Model Output Changes."
    European Conference on Computer Vision (ECCV), 2014.
    """
    
    def fetch_unlabelled(self, k):
        
        # Build list of candidate sample indices
        candidates = np.array([i for i in range(len(self.data)) \
                               if (i not in self.relevant_ids) \
                               and (i not in self.irrelevant_ids)])
        if len(candidates) < k:
            k = len(candidates)
        
        # Compute EMOC scores for all candidates
        scores = self.emoc_scores(candidates)
        
        # Return highest-scoring samples
        return candidates[np.argsort(scores)[::-1][:k]].tolist()
    
    
    def emoc_scores(self, ind):
        
        # Compute predictive mean and variance for all samples as length-r vectors
        mean, variance = self.gp.predict_stored(ind, cov_mode = 'diag')
        
        # Compute the model change for both possible labels and all candidates as u-by-(r+1) matrices
        k_diff = np.hstack((np.dot(self.gp.K_all[np.ix_(ind, self.gp.ind)], self.gp.K_inv), np.zeros((len(ind), 1)) - 1))
        denom = variance + self.gp.noise
        alpha_diff_pos = (( 1 - mean) / denom)[:,None] * k_diff
        alpha_diff_neg = ((-1 - mean) / denom)[:,None] * k_diff
        
        # Compute MOC (model output change) for all candidates and both possible labels as u-by-2 matrix
        moc = np.array([
            np.abs(np.dot(np.vstack((ad_pos, ad_neg)), self.gp.K_all[np.r_[self.gp.ind, [i]], :])).mean(axis = -1) \
            for i, ad_pos, ad_neg in zip(ind, alpha_diff_pos, alpha_diff_neg)
        ])
        
        # Compute EMOC (expected model output change) for all candidates as length-u vectors
        prob_neg = scipy.stats.norm.cdf(0, mean, np.sqrt(variance))
        prob_pos = 1 - prob_neg
        return prob_pos * moc[:,0] + prob_neg * moc[:,1]