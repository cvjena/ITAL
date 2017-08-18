import os.path
import configparser

import numpy as np

from italia import ITAL, RandomRetrieval, VarianceSampling, EMOC
from datasets import load_dataset


############
## Config ##
############


LEARNERS = {
    'ITAL'  : ITAL,
    'EMOC'  : EMOC,
    'random': RandomRetrieval,
    'var'   : VarianceSampling
}


def read_config_file(config_file, section, overrides):
    
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
    
    # Read config file
    config = read_config_file(config_file, section, overrides)
    
    # Set up dataset and learner
    dataset = config[section]['dataset']
    dataset = load_dataset(dataset, **config[dataset])
    learner = config[section]['method']
    learner = LEARNERS[learner](dataset.X_train_norm, **config[learner])
    
    return config, dataset, learner


def load_dataset_from_config(config_file, section, overrides = {}):
    
    # Read config file
    config = read_config_file(config_file, section, overrides)
    
    # Set up dataset
    dataset = config[section]['dataset']
    dataset = load_dataset(dataset, **config[dataset])
    return config, dataset



class ConversionInterpolation(configparser.BasicInterpolation):
    
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



##############
## Plotting ##
##############

def plot_data(data, relevance, query = None, retrieved = None, ax = None):
    
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
    
    import matplotlib.pyplot as plt
    
    retrieved = np.argsort(prob)[::-1][:k]
    
    fig, ax = plt.subplots(1, 2, figsize = (14, 5))
    plot_distribution(data, prob, query, ax = ax[0])
    plot_data(data, relevance, query, retrieved, ax = ax[1])
    plt.show()
