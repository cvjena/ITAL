import numpy as np
import scipy.stats

from .retrieval_base import ActiveRetrievalBase



class RandomRetrieval(ActiveRetrievalBase):
    
    def fetch_unlabelled(self, k):
        
        candidates = [i for i in range(len(self.data)) \
                      if (i not in self.relevant_ids) \
                      and (i not in self.irrelevant_ids)]
        
        return np.random.choice(candidates, min(k, len(candidates)), replace = False)



class VarianceSampling(ActiveRetrievalBase):
    
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



class EMOC(ActiveRetrievalBase):
    
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
        
        # Compute MOC (model output change) for all candidates and both possible labels as u-by-2 matrices
        moc = np.array([
            np.dot(np.vstack((ad_pos, ad_neg)), self.gp.K_all[np.r_[self.gp.ind, [i]], :]).mean(axis = -1) \
            for i, ad_pos, ad_neg in zip(ind, alpha_diff_pos, alpha_diff_neg)
        ])
        
        # Compute EMOC (expected model output change) for all candidates as length-u vectors
        prob_neg = scipy.stats.norm.cdf(0, mean, np.sqrt(variance))
        prob_pos = 1 - prob_neg
        return prob_pos * moc[:,0] + prob_neg * moc[:,1]