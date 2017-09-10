import numpy as np
import scipy.stats

import itertools
from tqdm import tqdm_notebook as tqdm, tnrange as trange
from multiprocessing import Pool

from .retrieval_base import ActiveRetrievalBase



class ITAL(ActiveRetrievalBase):
    """ Implementation of Information-theoretic Active Learning for Information Retrieval Applications. """
    
    def __init__(self, data, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 label_prob = 0.2, mistake_prob = 0.05,
                 monte_carlo_num_rel = None, monte_carlo_num_fb = None,
                 context_subset = None, parallelized = True):
         """
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - queries: list of query data points to initially fit the GP to. If empty, the GP will not be fitted.
        
        - length_scale: the `sigma` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - var: the `var` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - noise: the `sigma_noise` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - label_prob: the assumed probability that the user annotates a sample.
        
        - mistake_prob: the assumed probability that a label provided by the user is wrong.
        
        - monte_carlo_num_rel: if given, monte-carlo simultion is performed instead of full enumeration of all
                               possible relevance configurations for a given batch of samples. The number of
                               random samples drawn from the distribution is the size of the batch multiplied
                               with monte_carlo_num_rel.
        
        - monte_carlo_num_fb: if given, monte-carlo simultion is performed instead of full enumeration of all
                              possible feedback configurations for a given batch of samples. The number of
                              random samples drawn from the distribution is the size of the batch multiplied
                              with monte_carlo_num_fb.
        
        - context_subset: if given, candidate samples will be assumed to be correlated only with a maximum
                          number of context_subset other samples in the batch. This is a strong assumption
                          that can be used to fetch larger batches must faster, but at the cost of increased
                          redundancy and less diversity.
        
        - parallelized: if set to True, mutual information for a set of candidate samples will be computed
                        in parallel using multiprocessing.
        """
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.label_prob = label_prob
        self.mistake_prob = mistake_prob
        self.monte_carlo_num_rel = monte_carlo_num_rel
        self.monte_carlo_num_fb = monte_carlo_num_fb
        self.context_subset = context_subset
        self.parallelized = parallelized
    
    
    def fetch_unlabelled(self, k, show_progress = False):
        """ Fetches a batch of unlabelled samples to be annotated from the data matrix passed to __init__().
        
        # Arguments:
        
        - k: number of unlabelled samples to be chosen from the dataset.
        
        - show_progress: if set to True, a progress bar will be shown (requires the tqdm package).
        
        # Returns:
            list of indices of selected samples in the data matrix passed to __init__().
        """
        
        mutual_information = AppendedMutualInformation(self)
        candidates = [i for i in range(len(self.data)) \
                      if (i not in self.relevant_ids) \
                      and (i not in self.irrelevant_ids)]
        if len(candidates) < k:
            k = len(candidates)
        
        steps = trange(k) if show_progress else range(k)
        for it in steps:
            
            if self.parallelized and (it > 0):
                with Pool(initializer = _init_pool, initargs = (mutual_information,)) as p:
                    mi = p.map(_parallel_mi, candidates)
            else:
                mi = [mutual_information(i) for i in candidates]
            
            max_ind = np.argmax(mi)
            mutual_information.append(candidates[max_ind])
            del candidates[max_ind]
        
        return mutual_information.ret



class MutualInformation(object):
    """ Helper class for ITAL computing mutual information between the relevance distribution and the feedback distribution of a set of samples.
    
    MI(R,F|A) = sum(r) { sum(f) { p(R=r|A) * p(F=f|R=r,A) * log(p(R=r|F=f,A) / p(R=r|A)) } }
    """
    
    def __init__(self, learner, eps = 1e-12):
        """
        # Arguments:
        
        - learner: ITAL instance.
        
        - eps: small number added to denominators and logs.
        """
        
        self.learner = learner
        self.eps = eps
    
    
    def __call__(self, ret, mean = None, cov = None):
        """ Computes the mutual information between the relevance distribution and the feedback distribution of a set of samples.
        
        # Arguments:
        
        - ret: list of indices of samples in the candidate batch.
        
        - mean: optionally, pre-computed vector of predictive means for the samples.
                Will only be used if cov is given as well.
                If not given, will be obtained from ital.gp.GaussianProcess.predict_stored.
        
        - cov: optionally, pre-computed covariance matrix of the predictions.
               If not given, will be obtained from ital.gp.GaussianProcess.predict_stored.
        
        # Returns:
            float
        """
    
        if cov is None:
            mean, cov = self.learner.gp.predict_stored(ret, cov_mode = 'full')
        elif mean is None:
            mean = self.learner.gp.predict_stored(ret)
        
        mi = 0.0
        rel_iter, rel_mc_num = self.rel_iter(ret, mean, cov)
        for reli in rel_iter:

            rel = { ret[i] : r for i, r in enumerate(reli)}
            pr = self.prob_rel(reli, mean, cov)
            log_pr = np.log(pr + self.eps)

            fb_iter, fb_mc_num = self.fb_iter(ret, rel)
            for fbi in fb_iter:
                if any(fb != 0 for fb in fbi):

                    feedback = { ret[i] : fb for i, fb in enumerate(fbi) }
                    
                    pr_updated = self.updated_prob_rel(rel, feedback)
                    
                    cur_mi = np.log(pr_updated + self.eps) - log_pr
                    cur_mi *= self.likelihood(feedback, rel) if fb_mc_num == 0 else 1./fb_mc_num
                    if rel_mc_num == 0:
                        cur_mi *= pr
                    mi += cur_mi
        
        if rel_mc_num > 0:
            mi /= rel_mc_num
        
        return mi
    
    
    def rel_iter(self, ret, mean, cov):
        """ Creates an iterator over relevance configurations for a set of samples.
        
        # Arguments:
        
        - ret: list of indices of samples in the candidate batch.
        
        - mean: vector of predictive means of the samples.
        
        - cov: covariance matrix of the predictions.
        
        # Returns:
            iterator over boolean lists indicating relevance of the corresponding samples.
        """
        
        num = len(ret) * self.learner.monte_carlo_num_rel if self.learner.monte_carlo_num_rel is not None else None
        if (num is None) or (2 ** (len(ret) - 1) < num):
            return (itertools.product([False, True], repeat = len(ret)), 0)
        else:
            return (scipy.stats.multivariate_normal.rvs(mean, cov, num) > 0, num)
    
    
    def fb_iter(self, ret, rel):
        """ Creates an iterator over feedback configurations for a set of samples.
        
        # Arguments:
        
        - ret: list of indices of samples in the candidate batch.
        
        - rel: list of booleans indicating the true relevance of the samples.
        
        # Returns:
            iterator over lists with elements from the set [-1, 0, 1].
        """
        
        if (self.learner.label_prob >= 1) and (self.learner.mistake_prob <= 0):
            # Perfect user
            return ([[rel[r] for i, r in enumerate(ret)]], 1)
        
        elif self.learner.label_prob >= 1:
            # Maximally motivated user
            num = len(ret) * self.learner.monte_carlo_num_fb if self.learner.monte_carlo_num_fb is not None else None
            if (num is None) or (2 ** (len(ret) - 1) < num):
                return (itertools.product([-1, 1], repeat = len(ret)), 0)
            else:
                samples = np.random.choice([1, -1], (num, len(ret)), p = [
                        1.0 - self.learner.mistake_prob,
                        self.learner.mistake_prob
                ])
                samples[:,[i for i, r in enumerate(ret) if not rel[r]]] *= -1
                return (samples, num)
            
        else:
            # General case
            num = len(ret) * self.learner.monte_carlo_num_fb if self.learner.monte_carlo_num_fb is not None else None
            if (num is None) or (3 ** len(ret) < 2 * num):
                return (itertools.product([-1, 0, 1], repeat = len(ret)), 0)
            else:
                samples = np.random.choice([0, 1, -1], (num, len(ret)), p = [
                        1.0 - self.learner.label_prob,
                        self.learner.label_prob * (1.0 - self.learner.mistake_prob),
                        self.learner.label_prob * self.learner.mistake_prob
                ])
                samples[:,[i for i, r in enumerate(ret) if not rel[r]]] *= -1
                return (samples, num)
    
    
    def prob_rel(self, rel, mean, cov):
        """ Computes the probability P(R=r|A) of a given relevance configuration for a set of samples.
        
        # Arguments:
        
        - rel: list of booleans indicating the relevance of the samples.
        
        - mean: list of predictive means of the samples.
        
        - cov: covariance matrix of the predictions.
        
        # Returns:
            float
        """
        
        if len(rel) == 1:
            
            if rel[0]:
                return 1 - scipy.stats.norm.cdf(0, mean[0], np.sqrt(cov[0,0]))
            else:
                return scipy.stats.norm.cdf(0, mean[0], np.sqrt(cov[0,0]))
            
        else:
            
            stdev = np.sqrt(np.diag(cov))
            pivot = -mean / stdev
            infin = np.array(rel, dtype = int)

            i, j = np.tril_indices(cov.shape[0], -1)
            correl = cov[i, j] / (stdev[i] * stdev[j])

            err, pr, info = scipy.stats.mvn.mvndst(pivot, pivot, infin, correl,
                                                   maxpts = len(rel) * 100, abseps = 1e-4, releps = 1e-4)
            
            return pr
    
    
    def updated_prob_rel(self, rel, feedback):
        """ Computes the probability P(R=r|F=f,A) of a given relevance configuration for a set of samples, conditioned with the given feedback.
        
        # Arguments:
        
        - rel: list of booleans indicating the relevance of the samples.
        
        - feedback: the relevance feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples to a numeric value.
                    Values less than 0 indicate irrelevance, values greater than 0 indicate relevance, and
                    0 itself can be used to indicate that the user has not annotated the sample.
        
        # Returns:
            float
        """
        
        ret = sorted(rel.keys())
        mean, cov = self.learner.updated_prediction(feedback, ret)
        return self.prob_rel(np.array([rel[i] for i in ret]), mean, cov)
    
    
    def likelihood(self, feedback, rel):
        """ Computes the likelihood P(F=f|R=r,A) of given user feedback, given the true relevance of the samples.
        
        # Arguments:
        
        - feedback: the relevance feedback obtained from the user given as dictionary mapping the indices of
                    labelled samples to a numeric value.
                    Values less than 0 indicate irrelevance, values greater than 0 indicate relevance, and
                    0 itself can be used to indicate that the user has not annotated the sample.
        
        - rel: list of booleans indicating the true relevance of the samples.
        
        # Returns:
            float
        """

        if len(set(k for k, v in feedback.items() if v != 0) - set(rel.keys())) > 0:
            return 0

        prob = 1.0
        for i, r in rel.items():
            fb = feedback[i] if i in feedback else 0
            if fb == 0:
                prob *= 1.0 - self.learner.label_prob
            elif fb == 2 * r - 1:
                prob *= self.learner.label_prob * (1.0 - self.learner.mistake_prob)
            else:
                prob *= self.learner.label_prob * self.learner.mistake_prob
        return prob



class AppendedMutualInformation(MutualInformation):
    """ Helper class for ITAL computing mutual information between the relevance distribution and the feedback distribution of a set of samples.
    
    In contrast to MutualInformation, this class maintains a list of already selected samples and extends this list by the candidates.
    """
    
    def __init__(self, learner, ret=[]):
        """
        # Arguments:
        
        - learner: ITAL instance.
        
        - ret: list of indices of already selected samples.
        """
        
        MutualInformation.__init__(self, learner)
        self.set_ret(ret)
    
    
    def __call__(self, i):
        """ Computes the mutual information between the relevance distribution and the feedback distribution of the set of samples extended by a given one.
        
        # Arguments:
        - i: index of the new candidate sample.
        
        # Returns:
            float
        """
        
        if self.learner.context_subset and (self.learner.context_subset < len(self.ret)):
            context = np.random.choice(len(self.ret), self.learner.context_subset, replace = False)
            ctx_i = np.concatenate((context, [-1]))
            ret_ind = [self.ret[ci] for ci in context] + [i]
            return MutualInformation.__call__(self, ret_ind, self.learner.rel_mean[ret_ind], self.rel_covs[i][np.ix_(ctx_i, ctx_i)])
        else:
            return MutualInformation.__call__(self, self.ret + [i], self.learner.rel_mean[self.ret + [i]], self.rel_covs[i])
    
    
    def set_ret(self, ret):
        """ Changes the list of already selected samples.
        
        # Arguments:
        - ret: list of indices of already selected samples.
        """
        
        self.ret = [i for i in ret]
        self.rel_covs = self.learner.gp.predict_cov_batch(self.ret, np.arange(len(self.learner.data))) \
                        if len(self.ret) > 0 else self.learner.gp.predict_stored(cov_mode = 'diag')[1].reshape(-1, 1, 1)
    
    
    def append(self, i):
        """ Adds a sample to the list of selected ones.
        
        # Arguments:
        - i: index of the sample to be added.
        """
        
        self.ret.append(i)
        self.rel_covs = self.learner.gp.predict_cov_batch(self.ret, np.arange(len(self.learner.data)))



def _init_pool(mi):
    global _mi
    _mi = mi


def _parallel_mi(i):
    return _mi(i)