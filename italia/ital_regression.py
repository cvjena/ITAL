import numpy as np
import scipy.stats

import itertools
from tqdm import tqdm, trange
from multiprocessing import Pool

from .regression_base import ActiveRegressionBase



class ITAL_Regression(ActiveRegressionBase):
    """ Implementation of Information-theoretic Active Learning for Information Retrieval Applications using the KL divergence and marginal feedback probabilities. """
    
    def __init__(self, data = None, train_init = [], y_init = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 label_prob = 1.0,
                 change_estimation_subset = 0, monte_carlo_num = 1000,
                 parallelized = True):
        """
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - train_init: list of indices of initial training samples in data.
        
        - y_init: initial training values for the samples in train_init.
        
        - length_scale: the `sigma` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - var: the `var` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - noise: the `sigma_noise` hyper-parameter of the kernel (see documentation of italia.gp.GaussianProcess).
        
        - label_prob: the assumed probability that the user annotates a sample.
        
        - change_estimation_subset: If greater than 0, the model output change `KL(p(r|f,a), p(r|a))` will be estimated
                                    from this number of randomly selected additional samples and not only from the
                                    current batch. If set to `None`, the entire dataset will be used.
        
        - monte_carlo_num: Number of samples to draw from the prior value distribution for Monte-Carlo sampling of
                           feedback integral.
        
        - parallelized: if set to True, mutual information for a set of candidate samples will be computed
                        in parallel using multiprocessing.
        """
        
        ActiveRegressionBase.__init__(self, data, train_init, y_init, length_scale, var, noise)
        self.label_prob = label_prob
        self.change_estimation_subset = change_estimation_subset
        self.monte_carlo_num = monte_carlo_num
        self.parallelized = parallelized
    
    
    def fetch_unlabelled(self, k, show_progress = False):
        """ Fetches a batch of unlabelled samples to be annotated from the data matrix passed to __init__().
        
        # Arguments:
        
        - k: number of unlabelled samples to be chosen from the dataset.
        
        - show_progress: if set to True, a progress bar will be shown (requires the tqdm package).
        
        # Returns:
            list of indices of selected samples in the data matrix passed to __init__().
        """
        
        candidates = self.get_unseen()
        if len(candidates) < k:
            return candidates
        
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
    """ Helper class for ITAL_KL computing mutual information between the value distribution and the feedback distribution of a set of samples.
    
    \mathrm{MI}(R,F\ |\ A) = \int_{-\infty}^{\infty} { p(F=f\ |\ A) \cdot \mathrm{KL}\Bigl(p(R=r\ |\ F=f,A) \Bigm\| p(R=r\ |\ A)\Bigr) }
    """
    
    def __init__(self, learner, eps = 1e-6):
        """
        # Arguments:
        
        - learner: ITAL instance.
        
        - eps: small number added to denominators and logs.
        """
        
        self.learner = learner
        self.eps = eps
    
    
    def __call__(self, ret, fb_it = None, mean = None, cov = None):
        """ Computes the mutual information between the value distribution and the feedback distribution of a set of samples.
        
        # Arguments:
        
        - ret: list of indices of samples whose value distribution should be estimated.
        
        - fb_it: list of indices in ret to integrate feedback over. Setting this to None is equivalent to list(range(len(ret))).
        
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

        if fb_it is not None:
            mean_it = mean[fb_it]
            cov_it = cov[np.ix_(fb_it, fb_it)]
            val_ind = np.setdiff1d(np.arange(len(ret)), fb_it)
            ret_val = np.asarray(ret)[val_ind]
            mean_val = mean[val_ind]
            cov_val = cov[np.ix_(val_ind, val_ind)]
        else:
            mean_it = mean
            cov_it = cov
            ret_val = ret
            mean_val = mean
            cov_val = cov

        mi = 0.0
        for fbi in self.fb_iter(mean_it, cov_it, self.learner.monte_carlo_num):
            if any(fb is not None for fb in fbi):

                annotated_ret = [fb_it[i] if fb_it is not None else i for i, fb in enumerate(fbi) if fb is not None]

                mi += self.update_kl_divergence(
                    np.asarray(ret)[annotated_ret],
                    [fb for fb in fbi if fb is not None],
                    ret_val, mean_it, cov_it, cov_val
                )
        
        return mi / self.learner.monte_carlo_num
    
    
    def fb_iter(self, mean, cov, mc_num):
        """ Creates an iterator over randomly sampled feedback vectors for a certain set of samples.
        
        # Arguments:
        
        - mean: mean of the samples.
        
        - cov: covariance matrix of the samples.
        
        - mc_num: number of possible feedbacks to be drawn at random according to the feedback distribution.
        
        # Returns:
            iterator over lists with feedbacks for each sample. A feedback is either a floating-point number
            or None if the corresponding sample has not been annotated.
        """
        
        if self.learner.label_prob >= 1:
            # Maximally motivated user
            fb = scipy.stats.multivariate_normal.rvs(mean, cov + np.eye(cov.shape[0]) * self.eps, size = mc_num)
            if fb.ndim == 1:
                fb = fb[:,None]
            for fbi in fb:
                yield fbi
        else:
            # General case
            for annot in scipy.stats.bernoulli.rvs(self.learner.label_prob, size = (mc_num, len(mean))).astype(bool):
                fb = [None] * len(mean)
                annot_fb = scipy.stats.multivariate_normal.rvs(mean[annot], cov[np.ix_(annot, annot)] + np.eye(len(annot)) * self.eps)
                if annot_fb.ndim == 0:
                    annot_fb = annot_fb[None]
                i = 0
                for j, annotated in enumerate(annot):
                    if annotated:
                        fb[j] = annot_fb[i]
                        i += 1
                yield fb
    
    
    def update_kl_divergence(self, update_ind, y, pred_ind, update_mean, update_cov, pred_cov):
        """ Computes the KL divergence between the updated model p(r|f,a) and the current model p(r|a).
        
        # Arguments:
        
        - update_ind: list of indices of samples to be updated.
        
        - y: list of target values for the updated samples (either -1 or 1).
        
        - pred_ind: indices of samples whose updated distribution should be estimated.
        
        - mean: predictive mean of the samples before the update.
        
        - cov_inv: inverse of the predictive covariance matrix of the samples before the update.
        
        # Returns:
            float
        """
        
        mean_diff, cov_prod = self.learner.gp.updated_diff(update_ind, y, pred_ind, update_mean, update_cov, pred_cov)
        return (mean_diff + cov_prod.trace() - np.linalg.slogdet(cov_prod + np.eye(cov_prod.shape[0]) * self.eps)[1] - cov_prod.shape[0]) / 2.0



class AppendedMutualInformation(MutualInformation):
    """ Helper class for ITAL computing mutual information between the value distribution and the feedback distribution of a set of samples.
    
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
        """ Computes the mutual information between the value distribution and the feedback distribution of the set of samples extended by a given one.
        
        # Arguments:
        - i: index of the new candidate sample.
        
        # Returns:
            float
        """
        
        if self.learner._ce_subset is not None:
            
            if i in self.ce_subset_ind:
                ret = self.ce_subset_ext
                rel_it = self.ret_ind + [self.ce_subset_ind[i]]
                cov = self.cov_base
            else:
                ret = self.ce_subset_ext + [i]
                rel_it = self.ret_ind + [len(self.ce_subset_ext)]
                cov = self.covs[self.cov_ind[i]]
            
            return MutualInformation.__call__(self, ret, rel_it, mean = self.learner.mean[ret], cov = cov)
            
        else:
            
            return MutualInformation.__call__(self, self.ret + [i], mean = self.learner.mean[self.ret + [i]], cov = self.covs[i])
    
    
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
            self.covs = self.learner.gp.predict_cov_batch(self.ce_subset_ext, out_of_set)
            self.cov_base = self.learner.gp.predict_stored(self.ce_subset_ext, cov_mode = 'full')[1]
        
        else:
        
            self.covs = self.learner.gp.predict_cov_batch(self.ret, np.arange(len(self.learner.data))) \
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
                self.covs = self.learner.gp.predict_cov_batch(self.ce_subset_ext, out_of_set)
                self.cov_base = self.learner.gp.predict_stored(self.ce_subset_ext, cov_mode = 'full')[1]
            
            self.ret_ind.append(self.ce_subset_ind[i])
            
        else:
            
            self.covs = self.learner.gp.predict_cov_batch(self.ret, np.arange(len(self.learner.data)))



def _init_pool(mi):
    global _mi
    _mi = mi


def _parallel_mi(i):
    return _mi(i)