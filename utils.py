import os.path
import configparser
import math

import numpy as np

from italia import *
from datasets import load_dataset, RetrievalDataset, MultitaskRetrievalDataset, RegressionDataset


############
## Config ##
############


LEARNERS = {
    'ITAL'      : ITAL,
    'EMOC'      : EMOC,
    'MCMI'      : MCMI_min,
    'AdaptAL'   : AdaptAL,
    'SUD'       : SUD,
    'RBMAL'     : RBMAL,
    'TCAL'      : TCAL,
    'USDM'      : USDM,
    'entropy'   : EntropySampling,
    'random'    : RandomRetrieval,
    'border'    : BorderlineSampling,
    'border_div': BorderlineDiversitySampling,
    'topscoring': TopscoringSampling,
    'var'       : VarianceSampling,
    'unc'       : UncertaintySampling
}

REGRESSION_LEARNERS = {
    'ITAL'      : ITAL_Regression,
    'EMOC'      : EMOC_Regression,
    'entropy'   : EntropySampling_Regression,
    'random'    : RandomRetrieval_Regression,
    'var'       : VarianceSampling_Regression
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
        a (parser, dataset, learner) tuple whose individual components are:
        - parser: a configparser.ConfigParser instance,
        - dataset: a dataset.Dataset instance,
        - learner: either an italia.retrieval_base.ActiveRetrievalBase instance, an
                   italia.regression_base.ActiveRegressionBase instance.
                   The learner is usually initialized with the data from the dataset,
                   except in the case of a MultitaskDataset, where an uninitialized
                   learner will be returned.
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
    if isinstance(dataset, RegressionDataset):
        learner = REGRESSION_LEARNERS[learner](dataset.X_train_norm, **learner_config)
    elif isinstance(dataset, MultitaskRetrievalDataset):
        learner = LEARNERS[learner](**learner_config)
    else:
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


def area_under_curve(perf, normalized = True):
    """ Computes the area under curve for a sequence of performance metrics.
    
    # Arguments:
    
    - perf: either a vector of performance measures for a number of consecutive
            active learning steps or a 2-D array containing one such vector per row.
    
    - normalized: if True, the x-axis will be re-scaled so that the best possible AUC is 1.0.
    
    # Returns:
        float if perf is a vector, or vector with as many rows as perf if it is a 2-D array
    """
    
    perf = np.asarray(perf)
    if perf.ndim == 1:
        single = True
        perf = perf[None,:]
    else:
        single = False
    
    auc = (perf[:,1:-1].sum(axis = -1) + (perf[:,0] + perf[:,-1]) / 2) / perf.shape[1]
    
    return auc[0] if single else auc
