import sys, math
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from tqdm import tqdm, trange

import utils
from italia.gp import GaussianProcess



default_grid = OrderedDict((
    ('length_scale', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3., 4., 5., 6., 7., 8., 9., 10., 15., 20., 25.]),
    ('var', [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0]),
    ('noise', [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1])
))

default_init = { 'length_scale' : 0.1, 'var' : 1.0, 'noise' : 1e-6 }



def cross_validate_gp(dataset, relevance, gp_params, n_folds = 10):
    
    gp = GaussianProcess(dataset.X_train_norm, **gp_params)
    scores = np.ndarray((len(dataset.X_train),), dtype = float)
    
    kfold = StratifiedKFold(n_folds, shuffle = True, random_state = 0)
    for train_ind, test_ind in kfold.split(dataset.X_train_norm, relevance):
        gp.fit(train_ind, relevance[train_ind])
        scores[test_ind] = gp.predict_stored(test_ind)
    
    return average_precision_score(relevance, scores)


def optimize_gp_params(dataset, relevance, grid = default_grid, init = default_init, n_folds = 10, verbose = 1):
    
    param_names = list(grid.keys())
    cur_params = [init[p] for p in param_names]
    changing_param = 0
    perf = {}
    best_perf = 0
    
    while True:
        
        cur_perfs = {
            val : cross_validate_gp(dataset, relevance, { param_names[i] : val if i == changing_param else cur_params[i] for i in range(len(param_names)) }, n_folds = n_folds) \
            for val in grid[param_names[changing_param]]
        }
        best_val = max(cur_perfs.keys(), key = lambda v: cur_perfs[v])
        
        if cur_perfs[best_val] < best_perf:
            break
        best_perf = cur_perfs[best_val]
        
        if verbose > 0:
            print('{} : {:.4f}'.format(', '.join('{} = {}'.format(param_names[i], best_val if i == changing_param else cur_params[i]) for i in range(len(param_names))), best_perf))
        
        cur_params[changing_param] = best_val
        if tuple(cur_params) in perf:
            break
        else:
            perf[tuple(cur_params)] = best_perf
            changing_param = (changing_param + 1) % len(param_names)
    
    best_params = max(perf.keys(), key = lambda p: perf[p])
    return dict(zip(param_names, best_params)), best_perf



if __name__ == '__main__':
    
    # Parse arguments
    config_file = None
    overrides = {}
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            k, v = arg[2:].split('=', maxsplit = 1)
            overrides[k] = v
        elif config_file is None:
            config_file = arg
        else:
            print('Unexpected argument: {}'.format(arg))
            exit()
    if config_file is None:
        print('Usage: {} <experiment-config-file> [--<override-option>=<override-value> ...]'.format(sys.argv[0]))
        exit()
    
    # Load dataset
    config, dataset = utils.load_dataset_from_config(config_file, 'EXPERIMENT', overrides)
    query_classes = str(config.get('EXPERIMENT', 'query_classes', fallback = '')).split()
    if len(query_classes) == 0:
        query_classes = list(dataset.class_relevance.keys())
    else:
        for i in range(len(query_classes)):
            try:
                query_classes[i] = int(query_classes[i])
            except ValueError:
                pass
    
    # Optimize GP parameters individually for each class
    best_params = {}
    best_perf = {}
    for lbl in query_classes:
        print('--- CLASS {} ---'.format(lbl))
        relevance, _ = dataset.class_relevance[lbl]
        lbl_best, lbl_perf = optimize_gp_params(dataset, relevance, n_folds = config.getint('EXPERIMENT', 'n_folds', fallback = 10))
        best_params[lbl] = lbl_best
        best_perf[lbl] = lbl_perf
        print()
    
    # Print results
    for lbl in best_params.keys():
        print('Best parameters for class {} (AP: {:.2f}): {!r}'.format(lbl, best_perf[lbl], best_params[lbl]))
