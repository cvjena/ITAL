import numpy as np
import numexpr as ne
import scipy.spatial.distance
import scipy.linalg.lapack



def invh(M):
    """ Computes the inverse of a positive-semidefinite symmetric matrix.
    
    This uses the cholesky decomposition for inversion and is faster than the more generic np.linalg.inv.
    
    # Arguments:
    - M: the matrix to be inverted.
    
    # Returns:
        the inverse of M
    """
    
    if M.dtype == np.float32:
        potrf = scipy.linalg.lapack.spotrf
        potri = scipy.linalg.lapack.spotri
    else:
        potrf = scipy.linalg.lapack.dpotrf
        potri = scipy.linalg.lapack.dpotri
        if M.dtype != np.float64:
            M = M.astype(np.float64)
    
    zz , _ = potrf(M, False, False)
    inv_M , info = potri(zz)
    i, j = np.triu_indices_from(inv_M, k = 1)
    inv_M[j, i] = inv_M[i, j]
    return inv_M


def extend_inv(K, inv, inv_ind, new_ind, diag_noise = 0):
    """ Updates the inverse of a kernel matrix with new data.
    
    Let `K` be a square positive-definite matrix and `K1_inv = inv(K[inv_ind][:,inv_ind])` the inverse
    of a subset `K1` of this matrix `K`. This function computes the inverse `K2_inv = inv(K[inv_ind+new_ind][:,inv_ind+new_ind])`
    of an extension of `K1` with new data using the pre-computed `K1_inv`.
    
    
    # Arguments:
    
    - K: the full kernel matrix.
    
    - inv: the pre-computed inverse `K1_inv`.
    
    - inv_ind: vector of indices in `K` which define the subset of data present in `K1`.
    
    - new_ind: vector of indices to be added to the pre-computed inverse.
    
    - diag_noise: regularizer added to the diagonal.
    
    # Returns:
        `K2_inv`
    """

    ind = np.concatenate((inv_ind, new_ind))
    n, m, p = len(ind), len(inv_ind), len(new_ind)
    
    e = np.zeros((n, p))
    a = K[np.ix_(ind, new_ind)]
    for i in range(p):
        e[m+i, i] = 1
        a[m+i, i] += diag_noise - 1
    a[m:,:] /= 2

    U = np.hstack([a, e])
    V = np.vstack([e.T, a.T])

    inv_ext = np.vstack((
        np.hstack((inv, np.zeros((m, p)))),
        np.hstack((np.zeros((p, m)), np.eye(p)))
    ))
    return inv_ext - np.dot(
        np.dot(
            np.dot(inv_ext, U),
            np.linalg.inv(np.eye(2*p) + np.dot(np.dot(V, inv_ext), U))
        ),
        np.dot(V, inv_ext)
    )



class GaussianProcess(object):
    """ GP according to Eq. 2.23 and 2.24 in the GP bible.
    
    This implementation computes the kernel matrix of the entire data set once at
    the beginning to avoid redundant distance and kernel value computations.
    Thus, it is necessary to pass the entire data array to the constructor and
    referring to training and testing samples by their indices in that array.
    
    The kernel used by this GP is:
    `k(x_i, x_j) = var * exp(-||x_i - x_j||^2 / (2*sigma^2)) + sigma_noise * (i==j)`
    """
    
    def __init__(self, data, length_scale, var = 1.0, noise = 1e-6, pdist = None):
        """ Initializes the Gaussian Process.
        
        # Arguments:
        
        - data: entire dataset given as n-by-d array of n d-dimensional samples.
        
        - length_scale: the `sigma` hyper-parameter of the kernel.
        
        - var: the `var` hyper-parameter of the kernel.
        
        - noise: the `sigma_noise` hyper-parameter of the kernel.
        
        - pdist: optionally, pre-computed pair-wise distances of all samples may be passed to this argument
                 to avoid explicit distance computations. This could either be a 2-D square matrix containing
                 pair-wise squared euclidean distances or a 1-D array encoding the distances without
                 redundancy in the format as returned by `scipy.spatial.distance.pdist`.
        """
        
        self.X = np.array(data)
        self.length_scale = length_scale
        self.length_scale_sq = length_scale * length_scale
        self.var = var
        self.noise = noise
        
        self.K_all = self.kernel(self.X, self.X) if pdist is None else self.dist_kernel(pdist)
        self.reset()
    
    
    def reset(self):
        """ Resets the GP to its initial state directly after __init__. """
        
        self.ind = []
        self.y = self.K = self.K_inv = self.w = None
        self._inv_cache = {}
        self._cov_cache = {}
    
    
    def fit(self, ind, y):
        """ Fits the GP to a subset of the data passed to __init__.
        
        # Arguments:
        
        - ind: list of indices in the data matrix.
        
        - y: target values of the samples referred to by ind.
        
        # Returns:
            self
        """
        
        self.ind = [i for i in ind]
        self.y = np.array(y)
        self.K = self.K_all[np.ix_(self.ind, self.ind)] + self.noise * np.eye(len(self.ind))
        self.K_inv = invh(self.K)
        self.w = np.dot(self.K_inv, self.y)
        self._inv_cache = {}
        self._cov_cache = {}
        return self
    
    
    def update(self, ind, y):
        """ Updates the GP with new data (incremental fitting).
        
        If fit() has not been after the last call to __init__() or reset(),
        the effect of this method equals that of fit().
        
        # Arguments:
        
        - ind: list of indices in the data matrix.
        
        - y: target values of the samples referred to by ind.
        
        # Returns:
            self
        """
        
        if len(self.ind) == 0:
            return self.fit(ind, y)
        
        y = np.asarray(y)
        K_old_new = self.K_all[np.ix_(self.ind, ind)]
        K_new = self.K_all[np.ix_(ind, ind)] + self.noise * np.eye(len(ind))
        
        self.ind += [i for i in ind]
        self.y = np.concatenate((self.y, y))
        
        self.K = np.vstack((
            np.hstack((self.K, K_old_new)),
            np.hstack((K_old_new.T, K_new))
        ))
        self.K_inv = invh(self.K)
        
        self.w = np.dot(self.K_inv, self.y)
        
        self._inv_cache = {}
        self._cov_cache = {}
        return self
    
    
    def predict_stored(self, ind = None, cov_mode = None):
        """ Computes predictive mean and (optionally) variance or covariance for samples from the data matrix passed to __init__().
        
        # Arguments:
        
        - ind: list of n indices in the data matrix to make predictions for.
        
        - cov_mode: one of the following values:
            - None: only predict mean
            - 'diag': predict variance for each samples
            - 'full': predict a full covariance matrix for the given samples
        
        # Returns:
            - If cov_mode is None: a length-n vector of predictive means.
            - If cov_mode is 'diag': a tuple of two length-n vectors with predictive means and variances.
            - If cov_mode is 'full': a tuple of a length-n vector with predictive means and a n-by-n covariance matrix.
        """
        
        k_test = self.K_all[self.ind] if ind is None else self.K_all[np.ix_(self.ind, ind)]
        pred_mean = np.dot(self.w.T, k_test)
        
        if cov_mode == 'full':
            k_test_test = self.K_all if ind is None else self.K_all[np.ix_(ind, ind)]
            pred_cov = k_test_test - np.dot(k_test.T, np.dot(self.K_inv, k_test))
            return pred_mean, pred_cov
        elif cov_mode == 'diag':
            pred_var = np.maximum(0, self.var - np.sum(k_test * np.dot(self.K_inv, k_test), axis = 0))
            return pred_mean, pred_var
        else:
            return pred_mean
    
    
    def predict_cov_batch(self, base_ind, ind):
        """ Makes a batch of covariance predictions for a set of base samples extended by a single one of a batch of other samples.
        
        This is equivalent to `np.stack([predict_stored(np.concatenate([base_ind, [i]]), cov_mode='full')[1] for i in ind])`, but more efficient.
        
        # Arguments:
        
        - base_ind: list of n-1 indices of base samples in the data matrix.
        
        - ind: list of k indices of batch samples in the data matrix.
        
        # Returns:
            k-by-n-by-n array of covariance matrices.
        """
        
        k_base = self.K_all[np.ix_(self.ind, base_ind)]
        cov_base = self.K_all[np.ix_(base_ind, base_ind)] - np.dot(k_base.T, np.dot(self.K_inv, k_base))

        k_test = self.K_all[np.ix_(self.ind, ind)]
        var_test = self.var - np.sum(k_test * np.dot(self.K_inv, k_test), axis = 0)

        cov_base_test = self.K_all[np.ix_(base_ind, ind)] - np.dot(k_base.T, np.dot(self.K_inv, k_test))

        return np.concatenate((
                np.concatenate((np.tile(cov_base, (len(ind), 1, 1)), cov_base_test.T[:,:,None]), axis = 2),
                np.concatenate((cov_base_test.T[:,None,:], var_test[:,None,None]), axis = 2)
        ), axis = 1)
    
    
    def predict(self, X, cov_mode = None):
        """ Computes predictive mean and (optionally) variance or covariance for given samples.
        
        # Arguments:
        
        - X: n-by-d matrix of samples to predict mean and variance for.
        
        - cov_mode: one of the following values:
            - None: only predict mean
            - 'diag': predict variance for each samples
            - 'full': predict a full covariance matrix for the given samples
        
        # Returns:
            - If cov_mode is None: a length-n vector of predictive means.
            - If cov_mode is 'diag': a tuple of two length-n vectors with predictive means and variances.
            - If cov_mode is 'full': a tuple of a length-n vector with predictive means and a n-by-n covariance matrix.
        """
        
        k_test = self.kernel(X)
        pred_mean = np.dot(self.w.T, k_test)
        
        if cov_mode == 'full':
            pred_cov = self.kernel(X, X) - np.dot(k_test.T, np.dot(self.K_inv, k_test))
            return pred_mean, pred_cov
        elif cov_mode == 'diag':
            pred_var = np.maximum(0, self.var - np.sum(k_test * np.dot(self.K_inv, k_test), axis = 0))
            return pred_mean, pred_var
        else:
            return pred_mean
    
    
    def updated_prediction(self, ind, y, pred_ind, cov_mode = None):
        """ Obtains a prediction for a set of samples after an update of the GP with new data, but without actually updating it.
        
        # Arguments:
        
        - ind: list of indices of update samples in the data matrix.
        
        - y: target values of the samples referred to by ind.
        
        - pred_ind: list of n indices in the data matrix to make predictions for.
        
        - cov_mode: one of the following values:
            - None: only predict mean
            - 'diag': predict variance for each samples
            - 'full': predict a full covariance matrix for the given samples
        
        # Returns:
            - If cov_mode is None: a length-n vector of predictive means.
            - If cov_mode is 'diag': a tuple of two length-n vectors with predictive means and variances.
            - If cov_mode is 'full': a tuple of a length-n vector with predictive means and a n-by-n covariance matrix.
        """
        
        y = np.asarray(y)
        cache_key = tuple(ind)
        cov_cache_key = cache_key + (-1,) + tuple(pred_ind)
        
        if cache_key in self._inv_cache:
            K_inv = self._inv_cache[cache_key]
        else:
            K_inv = extend_inv(self.K_all, self.K_inv, self.ind, ind, self.noise)
            self._inv_cache[cache_key] = K_inv.copy()
        
        k_test = self.K_all[np.ix_(np.concatenate((self.ind, ind)), pred_ind)]
        pred_mean = np.dot(np.dot(K_inv, np.concatenate((self.y, y))).T, k_test)
        
        if cov_mode == 'full':
            if cov_cache_key in self._cov_cache:
                pred_cov = self._cov_cache[cov_cache_key]
            else:
                pred_cov = self.K_all[np.ix_(pred_ind, pred_ind)] - np.dot(k_test.T, np.dot(K_inv, k_test))
                self._cov_cache[cov_cache_key] = pred_cov.copy()
            return pred_mean, pred_cov
        elif cov_mode == 'diag':
            if cov_cache_key in self._cov_cache:
                pred_var = np.diag(self._cov_cache[cov_cache_key])
            else:
                pred_var = (self.var - np.sum(k_test * np.dot(K_inv, k_test), axis = 0))
            return pred_mean, np.maximum(0, pred_var)
        else:
            return pred_mean
    
    
    def updated_diff(self, ind, y, pred_ind, update_mean = None, update_cov = None, pred_cov = None):
        """ Computes the change of predictive mean and covariance after a simulated update with new data.
        
        # Arguments:
        
        - ind: list of indices of update samples in the data matrix.
        
        - y: target values of the samples referred to by ind.
        
        - pred_ind: list of n indices in the data matrix to make predictions for.
        
        - update_mean: predictive mean of the update samples. If not specified, will be obtained from `predict_stored()`.
        
        - update_cov: predictive covariance matrix of the update samples. If not specified, will be obtained from `predict_stored()`.
        
        - update_cov: predictive covariance matrix of the samples specified by `pred_ind` before the update. If not specified, will be obtained from `predict_stored()`.
        
        # Returns:
            tuple consisting of `(m - m')^T * S^(-1) * (m - m')` and `S^(-1) * S'`, where `m` and `m'` are the predictive means of the test samples before and after
            the update, respectively, and `S` and `S'` are the corresponding predictive covariances.
        """
        
        if update_cov is None:
            update_mean, update_cov = self.predict_stored(ind, cov_mode = 'full')
        elif update_mean is None:
            update_mean = self.predict_stored(ind)
        if pred_cov is None:
            _, pred_cov = self.predict_stored(pred_ind, cov_mode = 'full')
        
        y_diff = y - update_mean
        k_diff = np.linalg.multi_dot((
                    self.K_all[np.ix_(pred_ind, self.ind)],
                    self.K_inv,
                    self.K_all[np.ix_(self.ind, ind)]
                 )) - self.K_all[np.ix_(pred_ind, ind)]
        k_diff_norm = np.linalg.solve(update_cov + np.eye(update_cov.shape[0]) * self.noise, k_diff.T).T
        k_diff_inv = np.linalg.solve(pred_cov, k_diff_norm)

        mean_diff = np.linalg.multi_dot((y_diff.T, k_diff_norm.T, k_diff_inv, y_diff))
        cov_diff = np.eye(len(pred_ind)) - np.dot(k_diff_inv, k_diff.T)
        return mean_diff, cov_diff
    
    
    def kernel(self, a, b = None):
        """ Evaluates the kernel function of this GP.
        
        `kernel(X, Y)` with n-by-d and m-by-d data matrices X and Y will compute an
        n-by-m kernel matrix K with `K[i,j] = kernel(X[i], Y[i])`.
        
        `kernel(Y)` will compute `kernel(self.X, Y)`, where `self.X` refers to the
        subset of the data matrix passed to __init__() which the GP is currently
        fitted to.
        """
        
        if b is None:
            b, a = a, self.X[self.ind]
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim == 1:
            a = a[None,:]
        if b.ndim == 1:
            b = b[None,:]
        
        if a is b:
            norm = np.sum(a ** 2, axis = -1)
            return ne.evaluate('v * exp((A + B - 2 * C) / s)', { 'A' : norm[:,None], 'B' : norm[None,:], 'C' : np.dot(a, a.T), 's' : -2 * self.length_scale_sq, 'v' : self.var })
        else:
            a_norm = np.sum(a ** 2, axis = -1)
            b_norm = np.sum(b ** 2, axis = -1)
            return ne.evaluate('v * exp((A + B - 2 * C) / s)', { 'A' : a_norm[:,None], 'B' : b_norm[None,:], 'C' : np.dot(a, b.T), 's' : -2 * self.length_scale_sq, 'v' : self.var })
    
    
    def dist_kernel(self, dist):
        """ Evaluates the kernel function on a pre-computed pair-wise distance matrix.
        
        # Arguments:
        
        - dist: either a 2-D square matrix containing pair-wise squaired euclidean distances or a 1-D array encoding the
                distances without redundancy in the format as returned by `scipy.spatial.distance.pdist`.
        
        # Returns:
            square 2-D matrix with pair-wise kernel values
        """
        
        if dist.ndim == 1:
            sqexp = scipy.spatial.distance.squareform(ne.evaluate('v * exp(d / s)', { 'd' : dist, 's' : -2 * self.length_scale_sq, 'v' : self.var }))
            sqexp += np.diag(np.ones(sqexp.shape[0]) * self.var)
            return sqexp
        else:
            return ne.evaluate('v * exp(D / s)', { 'D' : dist, 's' : -2 * self.length_scale_sq, 'v' : self.var })
