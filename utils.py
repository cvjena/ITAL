import os.path
import configparser
import math

import numpy as np

from italia import *
from datasets import load_dataset


############
## Config ##
############


LEARNERS = {
    'ITAL'  : ITAL,
    'EMOC'  : EMOC,
    'random': RandomRetrieval,
    'border': BorderlineSampling,
    'var'   : VarianceSampling,
    'unc'   : UncertaintySampling
}


def read_config_file(config_file, section, overrides):
    """ Reads a configuration file.
    
    Config files follow the format understood by `configparser.ConfigParser`.
    In addition, this function tries to cast all values in the config file to int, float,
    or boolean. It also handles the special "import" option that can be used to specify
    a white-space separated list of other config files to be read before this one.
    
    # Arguments:
    
    - config_file: path to the config file.
    
    - section: name of the section to search for "import" options in and to apply overrides to.
    
    - overrides: dictionary with options overriding the ones read from the config file in the
                 section given by `section`.
    
    # Returns:
        a configparser.ConfigParser instance.
    """
    
    # Read config file
    config = configparser.ConfigParser(interpolation = ConversionInterpolation())
    with open(config_file) as cf:
        config.read_file(cf)
    
    # Handle imports of other config files
    imports = config.get(section, 'import', fallback = None)
    if imports:
        config_base = os.path.dirname(config_file)
        if config_base == '':
            config_base = '.'
        imports = [p.strip() if os.path.isabs(p.strip()) else os.path.join(config_base, p.strip()) for p in imports.split()]
        config.read(imports + [config_file])
    
    # Apply overrides
    for k, v in overrides.items():
        config[section][k] = v
    
    return config


def load_config(config_file, section, overrides = {}):
    """ Instantiates a dataset and a learner from a given config file.
    
    See `read_config_file` for information about the format of the config file
    and the arguments of this function.
    
    # Returns:
        a (configparser.ConfigParser, datasets.Dataset, italia.retrieval_base.ActiveRetrievalBase) tuple.
    """
    
    # Read config file
    config = read_config_file(config_file, section, overrides)
    
    # Set up dataset
    dataset = config[section]['dataset']
    dataset = load_dataset(dataset, **config[dataset])
    
    # Set up learner
    learner = config[section]['method']
    learner_config = dict(config['METHOD_DEFAULTS']) if 'METHOD_DEFAULTS' in config else dict()
    if learner in config:
        learner_config.update(config[learner])
    learner = LEARNERS[learner](dataset.X_train_norm, **learner_config)
    
    return config, dataset, learner


def load_dataset_from_config(config_file, section, overrides = {}):
    """ Instantiates a dataset from a given config file.
    
    See `read_config_file` for information about the format of the config file
    and the arguments of this function.
    
    # Returns:
        a (configparser.ConfigParser, datasets.Dataset) tuple.
    """
    
    # Read config file
    config = read_config_file(config_file, section, overrides)
    
    # Set up dataset
    dataset = config[section]['dataset']
    dataset = load_dataset(dataset, **config[dataset])
    return config, dataset



class ConversionInterpolation(configparser.BasicInterpolation):
    """ Interpolation for ConfigParser instances trying to cast all values to int, float, or boolean. """
    
    def before_get(self, parser, section, option, value, defaults):
        
        val = configparser.BasicInterpolation.before_get(self, parser, section, option, value, defaults)
        
        try:
            return int(val)
        except ValueError:
            pass
        
        try:
            return float(val)
        except ValueError:
            pass
        
        if val.lower() in ('yes','on','true'):
            return True
        elif val.lower() in ('no','off','false'):
            return False
        else:
            return val



########################
## Evaluation Metrics ##
########################

def ndcg(y_true, y_score):
    """ Computes the Normalized Discounted Cumulative Gain (NDCG) of given retrieval results.
    
    # Arguments:
    
    - y_true: ground-truth relevance labels of the retrieved samples.
    
    - y_score: predicted relevance scores of the retrieved samples.
    
    # Returns:
        float
    """
    
    num_relevant = sum(yt > 0 for yt in y_true)
    retrieved = np.argsort(y_score)[::-1]
    
    rank, cgain, normalizer = 0, 0.0, 0.0
    for ret in retrieved:
        rank += 1
        gain = 1.0 / math.log2(rank + 1)
        if y_true[ret] > 0:
            cgain += gain
        if rank <= num_relevant:
            normalizer += gain
    
    return cgain / normalizer



###################
## Visualization ##
###################

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
    
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt
    
    if retrieved is None:
        retrieved = []
    not_retrieved = np.setdiff1d(np.arange(len(data)), retrieved)
    
    colors = np.where(relevance == 1, 'b', 'gray')
    colors_ret = np.where(relevance == 1, 'c', 'orange')
    
    if ax == plt:
        plt.figure(figsize = (6.25, 5))
    ax.scatter(data[not_retrieved,0], data[not_retrieved,1], c = colors[not_retrieved], s = 15)
    if len(retrieved) > 0:
        ax.scatter(data[retrieved,0], data[retrieved,1], c = colors_ret[retrieved], s = 15)
    if query is not None:
        ax.scatter(*query, c = 'r', s = 15)
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
    
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt
    
    if ax == plt:
        plt.figure(figsize = (6.25, 5))
    prob_min, prob_max = prob.min(), prob.max()
    ax.scatter(data[:,0], data[:,1], c = plt.cm.viridis((prob - prob_min) / (prob_max - prob_min)), s = 15)
    if query is not None:
        ax.scatter(*query, c = 'r', s = 15)
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
    
    import matplotlib.pyplot as plt
    
    retrieved = np.argsort(prob)[::-1][:k]
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 5))
    plot_distribution(data, prob, query, ax = ax[0])
    plot_data(data, relevance, query, retrieved, ax = ax[1])
    plt.show()


def plot_learning_step(dataset, query, relevance, learner, ret, fb):
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
    
    - query: the index of the query image in dataset.X_train.
    
    - relevance: the ground-truth relevance labels of all samples in dataset.X_train.
    
    - learner: an italia.retrieval_base.ActiveRetrievalBase instance.
    
    - ret: the indices of the samples selected for the current active learning batch.
    
    - fb: a list of feedback provided for each sample in the current batch. Possible feedback values
          are -1 (irrelevant), 1 (relevant) or 0 (no feedback).
    """
    
    import matplotlib.pyplot as plt
    
    if dataset.imgs_train is not None:
    
        cols = max(10, len(ret))
        fig, axes = plt.subplots(6, cols, figsize = (cols, 6))
        axes[0,0].imshow(dataset.imgs_train[query], interpolation = 'bicubic', cmap = plt.cm.gray)
        for r, ax in zip(ret, axes[1]):
            ax.imshow(dataset.imgs_train[r], interpolation = 'bicubic', cmap = plt.cm.gray)
        top_ret = np.argsort(learner.rel_mean)[::-1][:cols*(len(axes)-2)]
        for r, ax in zip(top_ret, axes[2:].ravel()):
            ax.imshow(dataset.imgs_train[r], interpolation = 'bicubic', cmap = plt.cm.gray)
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
        plot_data(dataset.X_train, relevance, dataset.X_train[query], ret, axes[0,0])
        plot_data(dataset.X_train, relevance, dataset.X_train[query], [r for i, r in enumerate(ret) if fb[i] != 0], axes[0,1])
        plot_distribution(dataset.X_train, learner.rel_mean, dataset.X_train[query], axes[1,0])
        plot_data(dataset.X_train, relevance, dataset.X_train[query], np.argsort(learner.rel_mean)[::-1][:np.sum(relevance > 0)], axes[1,1])
        fig.tight_layout()
        plt.show()
    
    else:
    
        raise RuntimeError("Don't know how to plot this dataset.")
