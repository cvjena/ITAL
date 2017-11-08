import numpy as np
import scipy.stats, scipy.linalg, scipy.spatial.distance

import math
import itertools
from multiprocessing import Pool

from .retrieval_base import ActiveRetrievalBase
from .regression_base import ActiveRegressionBase



class RandomRetrieval(ActiveRetrievalBase):
    """ Selects samples at random. """
    
    def fetch_unlabelled(self, k):
        
        candidates = [i for i in range(len(self.data)) \
                      if (i not in self.relevant_ids) \
                      and (i not in self.irrelevant_ids)]
        
        return np.random.choice(candidates, min(k, len(candidates)), replace = False)


class RandomRetrieval_Regression(ActiveRegressionBase):
    """ Selects samples at random. """
    
    def fetch_unlabelled(self, k):
        
        candidates = [i for i in range(len(self.data)) \
                      if i not in self.labeled_ids]
        
        return np.random.choice(candidates, min(k, len(candidates)), replace = False)



class TopscoringSampling(ActiveRetrievalBase):
    """ Selects samples with maximum predictive mean. """
    
    def fetch_unlabelled(self, k):
        
        ranking = np.argsort(self.rel_mean)[::-1]
        ret = []
        for i in ranking:
            if (i not in self.relevant_ids) and (i not in self.irrelevant_ids):
                ret.append(i)
                if len(ret) >= k:
                    break
        return ret



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



class BorderlineDiversitySampling(ActiveRetrievalBase):
    """ Selects samples with small distance to the decision boundary while maintaining diversity among them w.r.t. their angle.
    
    Reference:
    Klaus Brinker.
    "Incorporating Diversity in Active Learning with Support Vector Machines."
    International Conference on Machine Learning (ICML), 2003.
    
    `alpha` is the linear combination coefficient interpolating between the distance and the diversity criterion,
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 alpha = 0.5):
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.alpha = alpha
    
    
    def fetch_unlabelled(self, k):
        
        candidates = [i for i in range(len(self.data)) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids)]
        
        # Select sample closest to the decision boundary as first sample
        min_ind = np.argmin(np.abs(self.rel_mean[candidates]))
        ret = [candidates[min_ind]]
        
        # Select more samples by minimizing a trade-off between distance to decision boundary and similarity to already selected samples
        for i in range(1, k):
            
            del candidates[min_ind]
            if len(candidates) == 0:
                break
            
            # Compute cosine similarity of candidates to selected samples in kernel space
            angle = self.gp.K_all[np.ix_(candidates, ret)]
            angle /= np.sqrt(self.gp.K_all[candidates, candidates])[:,None]
            angle /= np.sqrt(self.gp.K_all[ret, ret])[None,:]
            diversity = angle.max(axis = -1)
            
            # Select sample with minimum score
            scores = self.alpha * np.abs(self.rel_mean[candidates]) + (1.0 - self.alpha) * diversity
            min_ind = np.argmin(scores)
            ret.append(candidates[min_ind])
        
        return ret



class VarianceSampling(ActiveRetrievalBase):
    """ Selects samples with maximum predictive variance.
    
    If `use_correlations` is set to `True`, the covariance to other samples in the selected batch will also
    be taken into account by computing the score of a given batch of samples as the sum of their variance
    minus the sum of their covariance. Samples will be selected in a greedy fashion, starting with the one
    with the highest predictive variance and extending the batch successively.
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
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


class VarianceSampling_Regression(ActiveRegressionBase):
    """ Selects samples with maximum predictive variance.
    
    If `use_correlations` is set to `True`, the covariance to other samples in the selected batch will also
    be taken into account by computing the score of a given batch of samples as the sum of their variance
    minus the sum of their covariance. Samples will be selected in a greedy fashion, starting with the one
    with the highest predictive variance and extending the batch successively.
    """
    
    def __init__(self, data = None, train_init = [], y_init = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 use_correlations = False):
        
        ActiveRegressionBase.__init__(self, data, train_init, y_init, length_scale, var, noise)
        self.use_correlations = use_correlations
    
    
    def fetch_unlabelled(self, k):
        
        _, var = self.gp.predict_stored(cov_mode = 'diag')
        
        if self.use_correlations:
            
            ret = [max(range(var.size), key = lambda i: var[i] if i not in self.labeled_ids else 0)]
            for l in range(1, k):
                candidates = [i for i in range(var.size) if (i not in self.labeled_ids) and (i not in ret)]
                if len(candidates) == 0:
                    break
                covs = self.gp.predict_cov_batch(ret, candidates)
                ti, tj = np.tril_indices(covs.shape[1], -1)
                scores = np.diagonal(covs, 0, 1, 2).sum(axis = -1) - covs[:,ti,tj].sum(axis = -1)
                ret.append(candidates[np.argmax(scores)])
            
        else:
            
            ranking = np.argsort(var)[::-1]
            ret = []
            for i in ranking:
                if i not in self.labeled_ids:
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
        rel_var = rel_var[:len(self.data)]
        
        candidates = [i for i in range(rel_mean.size) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids)]
        max_ind = max(range(len(candidates)), key = lambda i: self.__class__.single_entropy(rel_mean[candidates[i]], rel_var[candidates[i]]))
        ret = [candidates[max_ind]]

        with Pool() as p:
            for l in range(1, k):
                del candidates[max_ind]
                if len(candidates) == 0:
                    break
                covs = self.gp.predict_cov_batch(ret, candidates)
                entropies = p.starmap(self.__class__.batch_entropy, [(rel_mean[ret+[candidates[i]]], covs[i]) for i in range(len(candidates))])
                max_ind = np.argmax(entropies)
                ret.append(candidates[max_ind])
        
        return ret
    
    
    @staticmethod
    def single_entropy(mean, var):
        
        prob_irr = max(1e-8, min(1.0 - 1e-8, scipy.stats.norm.cdf(0, mean, np.sqrt(var))))
        return -1 * (prob_irr * np.log(prob_irr) + (1.0 - prob_irr) * np.log(1.0 - prob_irr))
    
    
    @staticmethod
    def batch_entropy(mean, cov):
        
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


class EntropySampling_Regression(ActiveRegressionBase):
    """ Selects batches of samples with maximum entropy.
    
    Reference:
    Ksenia Konyushkova, Raphael Sznitman and Pascal Fua.
    "Geometry in Active Learning for Binary and Multi-class Image Segmentation."
    arXiv:1606.09029v2.
    
    For batch sampling, this implementation uses the joint distribution of the samples in the
    batch for computing the batch entropy.
    """
    
    def __init__(self, *args, **kwargs):
        
        ActiveRegressionBase.__init__(self, *args, **kwargs)
        self.constant = np.log(2 * np.pi * np.e)
    
    
    def fetch_unlabelled(self, k):
        
        _, var = self.gp.predict_stored(cov_mode = 'diag')
        
        candidates = [i for i in range(len(var)) if i not in self.labeled_ids]
        max_ind = max(range(len(candidates)), key = lambda i: self.single_entropy(var[candidates[i]]))
        ret = [candidates[max_ind]]

        for l in range(1, k):
            del candidates[max_ind]
            if len(candidates) == 0:
                break
            covs = self.gp.predict_cov_batch(ret, candidates)
            max_ind = max(range(len(candidates)), key = lambda i: self.batch_entropy(covs[i]))
            ret.append(candidates[max_ind])
        
        return ret
    
    
    def single_entropy(self, var):
        
        return (self.constant + np.log(max(var, 1e-8))) / 2
    
    
    def batch_entropy(self, cov):
        
        return (np.linalg.slogdet(cov + np.eye(cov.shape[0]) * 1e-8)[1] + cov.shape[0] * self.constant) / 2



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



class EMOC_Regression(ActiveRegressionBase):
    """ Selects samples with maximum expected model output change (EMOC). """
    
    def __init__(self, data = None, train_init = [], y_init = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 norm = 1):
        
        ActiveRegressionBase.__init__(self, data, train_init, y_init, length_scale, var, noise)
        self.norm = norm
    
    
    def fetch_unlabelled(self, k):
        
        # Build list of candidate sample indices
        candidates = np.array([i for i in range(len(self.data)) if (i not in self.labeled_ids)])
        if len(candidates) < k:
            k = len(candidates)
        
        # Compute EMOC scores for all candidates
        scores = self.emoc_scores(candidates)
        
        # Return highest-scoring samples
        return candidates[np.argsort(scores)[::-1][:k]].tolist()
    
    
    def emoc_scores(self, ind):
        
        emocScores = np.empty([len(ind)])
        muTilde = np.zeros([len(ind)])

        k = self.gp.K_all[np.ix_(self.gp.ind, ind)]

        _, sigmaF = self.gp.predict_stored(ind, cov_mode = 'diag')
        moments = self.gaussianAbsoluteMoment(muTilde, sigmaF)

        term1 = 1.0 / (sigmaF + self.gp.noise)

        preCalcMult = np.dot(np.linalg.solve(self.gp.K, k).T, self.gp.K_all[np.ix_(self.gp.ind, ind)])

        for idx in range(len(ind)):

            vAll = term1[idx] * (preCalcMult[idx,:] - self.gp.K_all[ind[idx],ind])
            emocScores[idx] = np.mean(np.power(np.abs(vAll), self.norm))

        return emocScores * moments
    
    
    def gaussianAbsoluteMoment(self, muTilde, predVar):

        f11 = scipy.special.hyp1f1(-0.5*self.norm, 0.5, -0.5*np.divide(muTilde**2,predVar))
        prefactors = ((2 * predVar**2)**(self.norm/2.0) * math.gamma((1 + self.norm)/2.0)) / np.sqrt(np.pi)

        return np.multiply(prefactors,f11)



class SUD(ActiveRetrievalBase):
    """ Sampling by Uncertainty and Density.
    
    Reference:
    Jingbo Zhu, Huizhen Wang, Tianshun Yao, and Benjamin K Tsou.
    "Active Learning with Sampling by Uncertainty and Density for Word Sense Disambiguation and Text Classification."
    International Conference on Computational Linguistics (COLING), 2008, pp. 1137-1144.
    
    The parameter `K` specifies the number of nearest neighbours to take into account for density computation.
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 K = 20):
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.K = K
    
    
    def fetch_unlabelled(self, k):
        
        candidates = [i for i in range(len(self.data)) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids)]
        
        # Compute uncertainty (entropy) for all candidates
        rel_mean, rel_var = self.gp.predict_stored(candidates, cov_mode = 'diag')
        irr_prob = np.maximum(1e-8, np.minimum(1.0 - 1e-8, scipy.stats.norm.cdf(0, rel_mean, np.sqrt(rel_var))))
        rel_prob = 1.0 - irr_prob
        unc = -1 * (rel_prob * np.log(rel_prob) + irr_prob * np.log(irr_prob))
        
        # Compute density for all candidates
        densities = (np.partition(scipy.spatial.distance.cdist(self.data[candidates], self.data, 'cosine'), self.K, axis = -1)[:,:self.K+1].sum(axis = -1) - 1.0) / self.K
        
        # Select samples with maximum product of uncertainty and density
        max_ind = np.argsort(unc * densities)[::-1]
        return [candidates[i] for i in max_ind[:k]]



class RBMAL(ActiveRetrievalBase):
    """ Ranked Batch-Mode Active Learning.
    
    Reference:
    Thiago N. C. Cardoso, Rodrigo M. Silva, Sérgio Canuto, Mirella M. Moro, and Marcos A Gonçalves.
    "Ranked batch-mode active learning."
    Information Sciences 379, 2017, pp. 313-337.
    """
    
    def fetch_unlabelled(self, k):
        
        # Compute relevance probabilities and uncertainties for all unlabeled samples
        rel_mean, rel_var = self.gp.predict_stored(cov_mode = 'diag')
        irr_prob = scipy.stats.norm.cdf(0, rel_mean, np.sqrt(rel_var))
        unc = 1.0 - np.abs(1.0 - 2 * irr_prob)
        
        # Greedily select samples maximizing a trade-off between uncertainty and similarity to already selected and training samples
        train_ids = list(self.relevant_ids | self.irrelevant_ids)
        candidates = [i for i in range(rel_mean.size) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids)]
        ret = []
        for l in range(1, k):
            
            # Compute similarity to already selected samples
            dist = scipy.spatial.distance.cdist(self.data[candidates], self.data[train_ids], 'cosine').min(axis = -1)
            
            # Compute combined score
            alpha = len(candidates) / (len(candidates) + len(train_ids) + len(ret))
            scores = alpha * dist + (1 - alpha) * unc[candidates]
            
            # Select sample with maximum score
            max_ind = np.argmax(scores)
            ret.append(candidates[max_ind])
            del candidates[max_ind]
            if len(candidates) == 0:
                break
        
        return ret



class TCAL(ActiveRetrievalBase):
    """ Triple Criteria Active Learning.
    
    Reference:
    Begüm Demir and Lorenzo Bruzzone.
    "A Novel Active Learning Method in Relevance Feedback for Content-Based Remote Sensing Image Retrieval"
    IEEE Transactions on Geoscience and Remote Sensing 53.5, 2015, pp. 2323-2333.
    
    This algorithm consist of two steps:
    1. Select the `m` samples closest to the decision boundary.
    2. Divide them into `k` clusters and from eache cluster, select the sample with the minimum average
       distance to all other samples in the cluster.
    
    The parameter `unc_factor` implicitly controls `m` by `m = unc_factor * k`.
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 unc_factor = 4):
        
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
        self.unc_factor = unc_factor
    
    
    def fetch_unlabelled(self, k):
        
        candidates = np.array([i for i in range(len(self.data)) if (i not in self.relevant_ids) and (i not in self.irrelevant_ids)])
        
        # Select sample closest to the decision
        m = self.unc_factor * k
        uncertain_ind = np.argpartition(np.abs(self.rel_mean[candidates]), m - 1)[:m]
        unc = candidates[uncertain_ind]
        
        # Divide them into k clusters
        from .external.kernel_kmeans import KernelKMeans
        succ = False
        while not succ:
            try:
                km = KernelKMeans(k, kernel = 'precomputed')
                km.fit(self.gp.K_all[np.ix_(unc, unc)])
                succ = True
            except ValueError:
                k -= 1
                if k == 0:
                    raise
        
        # Select the sample with the highest density from each cluster
        ret = []
        for i in range(k):
            cluster_ind = unc[km.labels_ == i]
            K_cluster = self.gp.K_all[np.ix_(cluster_ind, cluster_ind)]
            d_cluster = np.diag(K_cluster)
            densities = np.mean(d_cluster[:,None] + d_cluster[None,:] - 2 * K_cluster, axis = -1)
            ret.append(cluster_ind[np.argmin(densities)])
        
        return ret



class USDM(ActiveRetrievalBase):
    """ Uncertainty Sampling with Diversity Maximization.
    
    Reference:
    Yi Yang, Zhigang Ma, Feiping Nie, Xiaojun Chang, and Alexander G Hauptmann.
    "Multi-Class Active Learning by Uncertainty Sampling with Diversity Maximization."
    International Journal on Computer Vision, 2015, pp. 113-127.
    
    This is a translation of the MATLAB reference code provided by Xiaojun Chang at:
    http://www.cs.cmu.edu/~uqxchan1/codes/AL_semi_fast1.m
    """
    
    def __init__(self, data = None, queries = [], length_scale = 0.1, var = 1.0, noise = 1e-6,
                 knn = 5, r = 1.0, max_iter = 100, tol = 1e-6):
        
        self.knn = knn
        self.r = r
        self.max_iter = max_iter
        self.tol = tol
        ActiveRetrievalBase.__init__(self, data, queries, length_scale, var, noise)
    
    
    def fit(self, data, queries = []):
        
        ActiveRetrievalBase.fit(self, data, queries)
        
        if self.gp is not None:
            # Construct neighborhood matrix
            self.A = np.zeros_like(self.gp.K_all)
            neighbours = np.argpartition(np.diag(np.diag(self.gp.K_all)) - self.gp.K_all, self.knn, axis = -1)[:,:self.knn]
            row_ind = np.tile(np.arange(self.A.shape[0])[:,None], (1, self.knn))
            self.A[row_ind, neighbours] = 1
            self.A += 1e-6
            self.A = np.diag(self.A.sum(axis = -1)) - self.A
    
    
    def fetch_unlabelled(self, k):
        
        labeled_ind = np.array(list(self.relevant_ids | self.irrelevant_ids))
        if len(self.queries) > 0:
            labeled_ind = np.concatenate((labeled_ind, np.arange(len(self.data), len(self.data) + len(self.queries))))
        unlabeled_ind = np.setdiff1d(np.arange(len(self.data)), labeled_ind)
        
        # Compute class probabilities and negated entropy
        y = np.array([1. if i in self.relevant_ids else 0. for i in labeled_ind])
        prob = np.maximum(1e-8, np.minimum(1.0 - 1e-8, np.linalg.solve(-self.A[np.ix_(unlabeled_ind, unlabeled_ind)], np.dot(self.A[np.ix_(unlabeled_ind, labeled_ind)], y))))
        b = (self.r * (prob * np.log(prob) + (1.0 - prob) * np.log(1.0 - prob))) / np.log(0.5)
        
        # Compute ranking scores
        f = self._alm(self.gp.K_all[np.ix_(unlabeled_ind, unlabeled_ind)], b, k)
        
        return unlabeled_ind[np.argpartition(-f, k - 1)[:k]]
    
    
    def _alm(self, K, b, k):
        
        n = len(b)
        mu = 1e-6
        rho = 1.1
        f = np.ones(n) / n
        v = f.copy()
        lambda1 = 0.0
        lambda2 = np.zeros(n)
        obj = None
        
        for it in range(self.max_iter):
            
            A = K.copy()
            A += mu
            A[np.arange(A.shape[0]),np.arange(A.shape[0])] += mu
            e = mu * (v + np.ones(n)) - (lambda2 + lambda1 * np.ones(n)) - b
            f = np.linalg.solve(A, e)
            v = f + lambda2 / mu
            v[v < 0] = 0
            lambda1 += mu * (f.sum() - k)
            lambda2 += mu * (f - v)
            mu *= rho
            
            obj_prev = obj
            obj = np.dot(f, np.dot(K, f)) / 2 + np.dot(f, b)
            if (obj_prev is not None) and (abs(obj_prev - obj) < self.tol):
                break
        
        return f
