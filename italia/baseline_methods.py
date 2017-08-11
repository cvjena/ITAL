import numpy as np

from .retrieval_base import ActiveRetrievalBase



class RandomRetrieval(ActiveRetrievalBase):
    
    def fetch_unlabelled(self, k):
        
        candidates = [i for i in range(len(self.data)) \
                      if (i not in self.relevant_ids) \
                      and (i not in self.irrelevant_ids)]
        
        return np.random.choice(candidates, min(k, len(candidates)), replace = False)



class UncertaintySampling(ActiveRetrievalBase):
    
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