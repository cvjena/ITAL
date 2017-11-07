import os
import numpy as np
import matplotlib.pyplot as plt
import skimage, skimage.io
import tarfile



def plot_data(data, relevance, query = None, retrieved = None, ax = None):
    """ Plots a 2-dimensional dataset.
    
    # Arguments:
    
    - data: n-by-2 data array of n 2-dimensional samples.
    
    - relevance: vector of length n specifying the relevance of the samples
                 (entries equal to 1 are considered relevant).
    
    - query: optionally, a query vector with 2 elements.
    
    - retrieved: optionally, a list of indices of retrieved samples.
    
    - ax: the Axis instance to draw the plot on. If `None`, the global pyplot
          object will be used.
    """
    
    if ax is None:
        ax = plt
    
    if retrieved is None:
        retrieved = []
    not_retrieved = np.setdiff1d(np.arange(len(data)), retrieved)
    
    colors = np.where(np.asarray(relevance) == 1, 'b', 'gray')
    colors_ret = np.where(np.asarray(relevance) == 1, 'c', 'orange')
    
    if ax == plt:
        plt.figure(figsize = (6.25, 5))
    ax.scatter(data[not_retrieved,0], data[not_retrieved,1], c = colors[not_retrieved], s = 15)
    if len(retrieved) > 0:
        ax.scatter(data[retrieved,0], data[retrieved,1], c = colors_ret[retrieved], s = 15)
    if query is not None:
        query = np.asarray(query)
        if query.ndim == 1:
            query = query[None,:]
        ax.scatter(query[:,0], query[:,1], c = 'r', s = 15)
    if ax == plt:
        plt.show()


def plot_distribution(data, prob, query = None, ax = None):
    """ Plots the estimated relevance scores of a 2-dimensional dataset.
    
    # Arguments:
    
    - data: n-by-2 data array of n 2-dimensional samples.
    
    - prob: vector of length n containing the estimated relevance scores of the samples.
    
    - query: optionally, a query vector with 2 elements.
    
    - ax: the Axis instance to draw the plot on. If `None`, the global pyplot
          object will be used.
    """
    
    if ax is None:
        ax = plt
    
    if ax == plt:
        plt.figure(figsize = (6.25, 5))
    prob_min, prob_max = prob.min(), prob.max()
    ax.scatter(data[:,0], data[:,1], c = plt.cm.viridis((prob - prob_min) / (prob_max - prob_min)), s = 15)
    if query is not None:
        query = np.asarray(query)
        if query.ndim == 1:
            query = query[None,:]
        ax.scatter(query[:,0], query[:,1], c = 'r', s = 15)
    if ax == plt:
        plt.show()


def plot_dist_and_topk(data, relevance, prob, query = None, k = 25):
    """ Plots and shows the estimated relevance scores and the top k retrieved samples of a 2-dimensional dataset.
    
    # Arguments:
    
    - data: n-by-2 data array of n 2-dimensional samples.
    
    - relevance: Vector of length n specifying the relevance of the samples
                (entries equal to 1 are considered relevant).
    
    - prob: Vector of length n containing the estimated relevance scores of the samples.
    
    - query: Optionally, a query vector with 2 elements.
    
    - k: the number of top retrieved samples to be shown.
    """
    
    retrieved = np.argsort(prob)[::-1][:k]
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 5))
    plot_distribution(data, prob, query, ax = ax[0])
    plot_data(data, relevance, query, retrieved, ax = ax[1])
    plt.show()


def plot_learning_step(dataset, queries, relevance, learner, ret, fb):
    """ Plots and shows a single active learning step.
    
    The output of this function differs depending on the type of data:
    - If the dataset provides an `imgs_train` attribute, this function will plot the query image,
      the images in the current active learning batch, and the top few images retrieved using the
      current classifier.
    - If the dataset otherwise contains 2-dimensional data, this will show the 2-d plots of the
      current active learning batch, the samples annotated by the user, the estimated relevance
      scores of the entire dataset after updating the learner, and the top few samples retrieved
      using that updated classifier.
    
    # Arguments:
    
    - dataset: a datasets.Dataset instance.
    
    - queries: the index of the query image in dataset.X_train (may also be a list of query indices).
    
    - relevance: the ground-truth relevance labels of all samples in dataset.X_train.
    
    - learner: an italia.retrieval_base.ActiveRetrievalBase instance.
    
    - ret: the indices of the samples selected for the current active learning batch.
    
    - fb: a list of feedback provided for each sample in the current batch. Possible feedback values
          are -1 (irrelevant), 1 (relevant) or 0 (no feedback).
    """
    
    if isinstance(queries, int):
        queries = [queries]
    
    if dataset.imgs_train is not None:
    
        cols = max([10, len(queries), len(ret)])
        fig, axes = plt.subplots(6, cols, figsize = (cols, 6))
        for query, ax in zip(queries, axes[0]):
            ax.imshow(canonicalize_image(dataset.imgs_train[query]), interpolation = 'bicubic', cmap = plt.cm.gray)
            ax.set_xlabel(canonicalize_img_name(dataset.imgs_train[query]))
        for r, ax in zip(ret, axes[1]):
            ax.imshow(canonicalize_image(dataset.imgs_train[r]), interpolation = 'bicubic', cmap = plt.cm.gray)
            ax.set_xlabel(canonicalize_img_name(dataset.imgs_train[r]))
        top_ret = np.argsort(learner.rel_mean)[::-1][:cols*(len(axes)-2)]
        for r, ax in zip(top_ret, axes[2:].ravel()):
            ax.imshow(canonicalize_image(dataset.imgs_train[r]), interpolation = 'bicubic', cmap = plt.cm.gray)
            ax.set_xlabel(canonicalize_img_name(dataset.imgs_train[r]))
        for ax in axes.ravel():
            ax.axis('off')
        fig.tight_layout()
        plt.show()
    
    elif dataset.X_train.shape[1] == 2:
    
        fig, axes = plt.subplots(2, 2, figsize = (10, 7))
        axes[0,0].set_title('Active Learning Batch')
        axes[0,1].set_title('Labelled Examples')
        axes[1,0].set_title('Relevance Distribution')
        axes[1,1].set_title('Retrieval')
        plot_data(dataset.X_train, relevance, dataset.X_train[queries], ret, axes[0,0])
        plot_data(dataset.X_train, relevance, dataset.X_train[queries], [r for i, r in enumerate(ret) if fb[i] != 0], axes[0,1])
        plot_distribution(dataset.X_train, learner.rel_mean, dataset.X_train[queries], axes[1,0])
        plot_data(dataset.X_train, relevance, dataset.X_train[queries], np.argsort(learner.rel_mean)[::-1][:np.sum(relevance > 0)], axes[1,1])
        fig.tight_layout()
        plt.show()
    
    else:
    
        raise RuntimeError("Don't know how to plot this dataset.")


def plot_regression_step(dataset, init, learner, ret, fb):
    """ Plots and shows a single active regression step for 2-dimensional data.
    
    # Arguments:
    
    - dataset: a datasets.RegressionDataset instance.
    
    - init: list of indices of the initial training samples in dataset.X_train.
    
    - learner: an italia.regression_base.ActiveRegressionBase instance.
    
    - ret: the indices of the samples selected for the current active learning batch.
    
    - fb: a list of feedback provided for each sample in the current batch.
    """
    
    if isinstance(init, int):
        init = [init]
    
    if dataset.X_train.shape[1] == 2:
    
        fig, axes = plt.subplots(1, 3, figsize = (12, 4))
        axes[0].set_title('Active Learning Batch')
        axes[1].set_title('Labelled Examples')
        axes[2].set_title('Relevance Distribution')
        plot_data(dataset.X_train, [0] * len(dataset.X_train), dataset.X_train[init], ret, axes[0])
        plot_distribution(dataset.X_train, dataset.y_train, dataset.X_train[[r for i, r in enumerate(ret) if fb[i] is not None]], axes[1])
        plot_distribution(dataset.X_train, learner.mean, np.zeros((0,2)), axes[2])
        fig.tight_layout()
        plt.show()
    
    else:
    
        raise RuntimeError("Don't know how to plot this dataset.")


def canonicalize_image(img, color = True, channels_first = False):
    """ Converts an image to the canonical format, i.e., a `numpy.ndarray` with shape HxWx3, where the last axis represents RGB tuples with values in [0,1].
    
    If `color` is set to `False`, the last axis will be of size 1.
    
    If `channels_first` is set to True, the channel axis will be the first instead of the last axis.
    
    `img` can be one of the following:
        - a `HxW` `numpy.ndarray` giving a grayscale image,
        - a `HxWx3` `numpy.ndarray` giving a color image,
        - a string giving the filename of the image,
        - a tuple consisting of the path to a tarfile and either the name of the member or a corresponding `tarfile.TarInfo` object.
    """
    
    if isinstance(img, str):
        img = skimage.io.imread(img, as_grey = not color, img_num = 0)
    elif (isinstance(img, tuple) or isinstance(img, list)) and (len(img) == 2):
        with tarfile.open(img[0]) as tf:
            img = skimage.io.imread(tf.extractfile(img[1]), as_grey = not color, img_num = 0)
    
    img = skimage.img_as_float(img).astype(np.float32, copy = False)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    if (not color) and (img.shape[2] == 3):
        img = np.mean(img, axis = 2, keepdims = True)
    
    if channels_first:
        img = np.transpose(img, (2, 0, 1))
    
    return img


def canonicalize_img_name(img):
    
    if isinstance(img, str):
        return os.path.splitext(os.path.basename(str))[0]
    elif (isinstance(img, tuple) or isinstance(img, list)) and (len(img) == 2) and isinstance(img[1], str):
        return os.path.splitext(img[1])[0]