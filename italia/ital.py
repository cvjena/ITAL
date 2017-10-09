import numpy as np
import scipy.stats

import itertools
from tqdm import tqdm, trange
from multiprocessing import Pool

from .retrieval_base import ActiveRetrievalBase



class ITAL(ActiveRetrievalBase):
    """ Implementation of Information-theoretic Active Learning for Information Retrieval Applications. """
    
    def __init__(self, data, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 label_prob = 1.0, mistake_prob = 0.0,
                 change_estimation_subset = 0, clip_cov = 0,
                 monte_carlo_num_rel = None, monte_carlo_num_fb = None,
                 parallelized = True):
        """
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - queries: list of query data points to initially fit the GP to. If empty, the GP will not be fitted.
        
        - length_scale: the `sigma` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - var: the `var` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - noise: the `sigma_noise` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - label_prob: the assumed probability that the user annotates a sample.
        
        - mistake_prob: the assumed probability that a label provided by the user is wrong.
        
        - change_estimation_subset: If greater than 0, the model output change `p(r|f,a)/p(r|a)` will be estimated
                                    from this number of randomly selected additional samples and not only from the
                                    current batch. If set to `None`, the entire dataset will be used.
        
        - clip_cov: Correlations between samples whose absolute correlation coefficient is below this threshold
                    will be discarded so that the corresponding pairs of samples are assumed to be independent.
                    This can be used to decompose high-dimensional distributions into smaller factors to speed
                    up approximation of joint cumulative distribution functions.
        
        - monte_carlo_num_rel: if given, monte-carlo simultion is performed instead of full enumeration of all
                               possible relevance configurations for a given batch of samples. The number of
                               random samples drawn from the distribution is the size of the batch multiplied
                               with monte_carlo_num_rel.
        
        - monte_carlo_num_fb: if given, monte-carlo simultion is performed instead of full enumeration of all
                              possible feedback configurations for a given batch of samples. The number of
                              random samples drawn from the distribution is the size of the batch multiplied
                              with monte_carlo_num_fb.
        
        - parallelized: if set to True, mutual information for a set of candidate samples will be computed
                        in parallel using multiprocessing.
        """
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.label_prob = label_prob
        self.mistake_prob = mistake_prob
        self.change_estimation_subset = change_estimation_subset
        self.clip_cov = clip_cov
        self.monte_carlo_num_rel = monte_carlo_num_rel
        self.monte_carlo_num_fb = monte_carlo_num_fb
        self.parallelized = parallelized
    
    
    def fetch_unlabelled(self, k, show_progress = False):
        """ Fetches a batch of unlabelled samples to be annotated from the data matrix passed to __init__().
        
        # Arguments:
        
        - k: number of unlabelled samples to be chosen from the dataset.
        
        - show_progress: if set to True, a progress bar will be shown (requires the tqdm package).
        
        # Returns:
            list of indices of selected samples in the data matrix passed to __init__().
        """
        
        candidates = [i for i in range(len(self.data)) \
                      if (i not in self.relevant_ids) \
                      and (i not in self.irrelevant_ids)]
        if len(candidates) < k:
            k = len(candidates)
        
        if self.change_estimation_subset is None:
            self._ce_subset = candidates
        elif self.change_estimation_subset > 0:
            self._ce_subset = sorted(np.random.choice(candidates, min(len(candidates), self.change_estimation_subset), replace = False))
        else:
            self._ce_subset = None
        
        mutual_information = AppendedMutualInformation(self)
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
    
    
    def __call__(self, ret, rel_it = None, mean = None, cov = None):
        """ Computes the mutual information between the relevance distribution and the feedback distribution of a set of samples.
        
        # Arguments:
        
        - ret: list of indices of samples in the candidate batch.
        
        - rel_it: list of indices in ret to integrate over. Setting this to None is equivalent to list(range(len(ret))).
        
        - mean: optionally, pre-computed vector of predictive means for the samples.
                Will only be used if cov is given as well.
                If not given, will be obtained from ital.gp.GaussianProcess.predict_stored.
        
        - cov: optionally, pre-computed covariance matrix of the predictions.
               If not given, will be obtained from ital.gp.GaussianProcess.predict_stored.
        
        # Returns:
            float
        """
    
        if rel_it is None:
            return self._call_iter_all(ret, mean=mean, cov=cov)
        else:
            return self._call_iter_sub(ret, rel_it, mean=mean, cov=cov)
    
    
    def _call_iter_all(self, ret, mean = None, cov = None):
        """ Handles __call__ for rel_it == None. """
        
        if cov is None:
            mean, cov = self.learner.gp.predict_stored(ret, cov_mode = 'full')
        elif mean is None:
            mean = self.learner.gp.predict_stored(ret)
        
        mi = 0.0
        rel_iter, rel_mc_num = self.rel_iter(ret, mean, cov)
        for reli in rel_iter:

            rel = { ret[i] : r for i, r in enumerate(reli) }
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
    
    
    def _call_iter_sub(self, ret, rel_it, mean = None, cov = None):
        """ Handles __call__ for rel_it != None. """
        
        if cov is None:
            mean, cov = self.learner.gp.predict_stored(ret, cov_mode = 'full')
        elif mean is None:
            mean = self.learner.gp.predict_stored(ret)
        
        mean_it = mean[rel_it]
        cov_it = cov[np.ix_(rel_it, rel_it)]
        
        rel_vec = (mean > 0)
        rel = { i : r for i, r in zip(ret, rel_vec) }
        
        mi = 0.0
        rel_iter, rel_mc_num = self.rel_iter(np.asarray(ret)[rel_it], mean_it, cov_it)
        for reli in rel_iter:

            rel_sub = {}
            for i, r in zip(rel_it, reli):
                rel_vec[i] = r
                rel[ret[i]] = r
                rel_sub[ret[i]] = r
            
            pr = self.prob_rel(reli, mean_it, cov_it)
            log_pr = np.log(self.prob_rel(rel_vec, mean, cov) + self.eps)

            fb_iter, fb_mc_num = self.fb_iter(np.asarray(ret)[rel_it], rel)
            for fbi in fb_iter:
                if any(fb != 0 for fb in fbi):

                    feedback = { ret[i] : fb for i, fb in zip(rel_it, fbi) }
                    
                    pr_updated = self.updated_prob_rel(rel, feedback)
                    
                    cur_mi = np.log(pr_updated + self.eps) - log_pr
                    cur_mi *= self.likelihood(feedback, rel_sub) if fb_mc_num == 0 else 1./fb_mc_num
                    if rel_mc_num == 0:
                        cur_mi *= pr
                    mi += cur_mi
            
            for i in rel_it:
                rel_vec[i] = (mean[i] > 0)
                rel[ret[i]] = rel_vec[i]
        
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
        
        - rel: dictionary mapping sample indices to their true relevance.
        
        # Returns:
            iterator over lists with elements from the set [-1, 0, 1].
        """
        
        if (self.learner.label_prob >= 1) and (self.learner.mistake_prob <= 0):
            # Perfect user
            return ([[1 if rel[r] else -1 for r in ret]], 1)
        
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
        
        if (len(rel) > 5) and (self.learner.clip_cov > 0) and (self.learner.clip_cov < 1):
        
            return self._grouped_prob_rel(rel, mean, cov, self.learner.clip_cov)
        
        elif len(rel) == 1:
            
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
    
    
    def _grouped_prob_rel(self, rel, mean, cov, clip_th):
        """ Approximates the probability P(R=r|A) of a given relevance configuration using a factorization of the covariance matrix after thresholding.
        
        Any correlations between samples with a correlation coefficient less than clip_th will be discarded.
        This sparsifies the covariance matrix which can then often be decomposed into smaller distributions for speeding up the computation.
        """
    
        stdev = np.sqrt(np.diag(cov))
        corr = cov / (stdev[:,None] * stdev[None,:])
        pivot = -mean / stdev
        infin = np.array(rel, dtype = int)

        groups = group_cov(corr, clip_th)
        
        if len(groups) == 1:
            
            i, j = np.tril_indices(corr.shape[0], -1)
            correl = corr[i,j]

            err, pr, info = scipy.stats.mvn.mvndst(pivot, pivot, infin, correl,
                                                   maxpts = len(rel) * 100, abseps = 1e-4, releps = 1e-4)
            return pr
        
        else:

            prob = 1
            for g in groups:

                if len(g) == 1:

                    pr = scipy.stats.norm.cdf(0, mean[g[0]], stdev[g[0]])
                    prob *= pr if not rel[g[0]] else 1 - pr

                else:

                    group_corr = corr[np.ix_(g, g)]
                    i, j = np.tril_indices(group_corr.shape[0], -1)
                    correl = group_corr[i,j]

                    err, pr, info = scipy.stats.mvn.mvndst(pivot[g], pivot[g], infin[g], correl,
                                                           maxpts = len(g) * 100, abseps = 1e-4, releps = 1e-4)
                    prob *= pr

            return prob
    
    
    def updated_prob_rel(self, rel, feedback):
        """ Computes the probability P(R=r|F=f,A) of a given relevance configuration for a set of samples, conditioned with the given feedback.
        
        # Arguments:
        
        - rel: dictionary mapping sample indices to their assumed relevance.
        
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
        
        if self.learner._ce_subset is not None:
            
            if i in self.ce_subset_ind:
                ret = self.ce_subset_ext
                rel_it = self.ret_ind + [self.ce_subset_ind[i]]
                cov = self.rel_cov_base
            else:
                ret = self.ce_subset_ext + [i]
                rel_it = self.ret_ind + [len(self.ce_subset_ext)]
                cov = self.rel_covs[self.cov_ind[i]]
            
            return MutualInformation.__call__(self, ret, rel_it, mean = self.learner.rel_mean[ret], cov = cov)
            
        else:
            
            return MutualInformation.__call__(self, self.ret + [i], mean = self.learner.rel_mean[self.ret + [i]], cov = self.rel_covs[i])
    
    
    def set_ret(self, ret):
        """ Changes the list of already selected samples.
        
        # Arguments:
        - ret: list of indices of already selected samples.
        """
        
        self.ret = [i for i in ret]
        
        if self.learner._ce_subset is not None:
            
            self.ce_subset_ind = { ind : i for i, ind in enumerate(self.learner._ce_subset) }   # maps sample indices to indices in self.ce_subset_ext
            self.ce_subset_ext = [i for i in self.ret if i not in self.ce_subset_ind]           # indices of change estimation subset + indices of current batch
            for i, ind in enumerate(self.ce_subset_ext):
                self.ce_subset_ind[ind] = len(self.learner._ce_subset) + i
            self.ce_subset_ext = self.learner._ce_subset + self.ce_subset_ext
            self.ret_ind = [self.ce_subset_ind[i] for i in self.ret]    # indices of samples in current batch in self.ce_subset_ext
            
            out_of_set = np.setdiff1d(np.arange(len(self.learner.data)), self.ce_subset_ext)
            self.cov_ind = { ind : i for i, ind in enumerate(out_of_set) }
            self.rel_covs = self.learner.gp.predict_cov_batch(self.ce_subset_ext, out_of_set)
            self.rel_cov_base = self.learner.gp.predict_stored(self.ce_subset_ext, cov_mode = 'full')[1]
        
        else:
        
            self.rel_covs = self.learner.gp.predict_cov_batch(self.ret, np.arange(len(self.learner.data))) \
                            if len(self.ret) > 0 else self.learner.gp.predict_stored(cov_mode = 'diag')[1].reshape(-1, 1, 1)
    
    
    def append(self, i):
        """ Adds a sample to the list of selected ones.
        
        # Arguments:
        - i: index of the sample to be added.
        """
        
        self.ret.append(i)
        
        if self.learner._ce_subset is not None:
            
            if i not in self.ce_subset_ext:
                
                self.ce_subset_ind[i] = len(self.ce_subset_ext)
                self.ce_subset_ext.append(i)
                
                out_of_set = np.setdiff1d(np.arange(len(self.learner.data)), self.ce_subset_ext)
                self.cov_ind = { ind : i for i, ind in enumerate(out_of_set) }
                self.rel_covs = self.learner.gp.predict_cov_batch(self.ce_subset_ext, out_of_set)
                self.rel_cov_base = self.learner.gp.predict_stored(self.ce_subset_ext, cov_mode = 'full')[1]
            
            self.ret_ind.append(self.ce_subset_ind[i])
            
        else:
            
            self.rel_covs = self.learner.gp.predict_cov_batch(self.ret, np.arange(len(self.learner.data)))



def group_cov(cov, th = 1e-3):
    """ Decomposes a covariance or correlation matrix into smaller groups.
    
    # Arguments:
    
    - cov: the covariance or correlation matrix.
    
    - th: any entries in the matrix less than this threshold will be discarded.
    
    # Returns:
        a list of lists of indices defining mutually independent groups of samples.
    """
    
    clipped_cov = np.abs(cov) > th
    unassigned = np.arange(cov.shape[0])
    groups = []
    
    while len(unassigned) > 0:
        group = np.array([], dtype=int)
        new_members = np.nonzero(clipped_cov[unassigned[0]])[0]
        while len(new_members) > 0:
            group = np.concatenate([group, new_members])
            unassigned = np.setdiff1d(unassigned, new_members)
            new_members = np.intersect1d(np.nonzero(clipped_cov[group].max(axis = 0))[0], unassigned, assume_unique = True)
        groups.append(group)
    
    return groups


def _init_pool(mi):
    global _mi
    _mi = mi


def _parallel_mi(i):
    return _mi(i)