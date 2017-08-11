import numpy as np

from .gp import GaussianProcess



class ActiveRetrievalBase(object):
    
    def __init__(self, data, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6):
        
        self.data = data
        self.queries = queries
        self.gp = GaussianProcess(
            np.concatenate((self.data, self.queries)) if len(self.queries) > 0 else self.data,
            length_scale, var, noise
        )
        
        self.reset()
    
    
    def reset(self):
        
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
        
        ind = np.argsort(self.rel_mean)[::-1]
        return ind[:k] if k is not None else ind
    
    
    def fetch_unlabelled(self, k):
        
        raise NotImplementedError('fetch_unlabelled() has to be implemented in a derived class.')
    
    
    def update(self, feedback):
        
        rel, irr = self.partition_feedback(feedback)
        if len(rel) + len(irr) > 0:
        
            self.gp.update(rel + irr, np.concatenate((np.ones(len(rel)), -1 * np.ones(len(irr)))))
            self.rel_mean = self.gp.predict_stored()[:len(self.data)]

            self.relevant_ids.update(rel)
            self.irrelevant_ids.update(irr)
            self.rounds += 1
    
    
    def updated_prediction(self, feedback, test_ind, cov_mode = 'full'):
        
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