import sys, math
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import average_precision_score, mean_squared_error
import scipy.spatial
from tqdm import tqdm, trange

import utils
from datasets import RegressionDataset, MultitaskRetrievalDataset
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
    
    kfold = StratifiedKFold(n_folds, shuffle = True, random_state = 0) if relevance is not None else KFold(n_folds, shuffle = True, random_state = 0)
    for train_ind, test_ind in kfold.split(dataset.X_train_norm, relevance):
        gp.fit(train_ind, relevance[train_ind] if relevance is not None else dataset.y_train[train_ind])
        scores[test_ind] = gp.predict_stored(test_ind)
    
    return average_precision_score(relevance, scores) if relevance is not None else -math.sqrt(mean_squared_error(dataset.y_train, scores))


def cross_validate_fewshot(dataset, relevance, gp_params, n_folds = 10):
    
    gp = GaussianProcess(dataset.X_train_norm, **gp_params)
    perf = []
    
    kfold = StratifiedKFold(n_folds, shuffle = True, random_state = 0) if relevance is not None else KFold(n_folds, shuffle = True, random_state = 0)
    for train_ind, test_ind in kfold.split(dataset.X_train_norm, relevance):
        gp.fit(test_ind, relevance[test_ind] if relevance is not None else dataset.y_train[train_ind])
        scores = gp.predict_stored(train_ind)
        perf.append(average_precision_score(relevance[train_ind], scores) if relevance is not None else -math.sqrt(mean_squared_error(dataset.y_train[train_ind], scores)))
    
    return np.mean(perf)


def optimize_gp_params(dataset, relevance, grid = default_grid, init = default_init, n_folds = 10, fewshot = False, verbose = 1):
    
    param_names = list(grid.keys())
    cur_params = [init[p] for p in param_names]
    changed = [True] * len(param_names)
    changing_param = 0
    perf = {}
    best_perf = -np.infty
    
    pdist = scipy.spatial.distance.pdist(dataset.X_train_norm, 'sqeuclidean')
    
    while any(changed):
        
        cur_perfs = {}
        for val in grid[param_names[changing_param]]:
            cv_params ={ param_names[i] : val if i == changing_param else cur_params[i] for i in range(len(param_names)) }
            cv_params['pdist'] = pdist
            if fewshot:
                cur_perfs[val] = cross_validate_fewshot(dataset, relevance, cv_params, n_folds = n_folds)
            else:
                cur_perfs[val] = cross_validate_gp(dataset, relevance, cv_params, n_folds = n_folds)
            if verbose > 1:
                print('    {} = {} : {:.4f}'.format(param_names[changing_param], val, cur_perfs[val]))
        best_val = max(cur_perfs.keys(), key = lambda v: cur_perfs[v])
        
        if cur_perfs[best_val] < best_perf:
            break
        best_perf = cur_perfs[best_val]
        
        if verbose > 0:
            print('{} : {:.4f}'.format(', '.join('{} = {}'.format(param_names[i], best_val if i == changing_param else cur_params[i]) for i in range(len(param_names))), best_perf))
        
        changed[changing_param] = (best_val != cur_params[changing_param])
        cur_params[changing_param] = best_val
        perf[tuple(cur_params)] = best_perf
        changing_param = (changing_param + 1) % len(param_names)
    
    best_params = max(perf.keys(), key = lambda p: perf[p])
    return dict(zip(param_names, best_params)), best_perf if relevance is not None else -best_perf



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
    is_regression = isinstance(dataset, RegressionDataset)
    if is_regression:
    
        best_params, best_perf = optimize_gp_params(dataset, None,
                                                    n_folds = config.getint('EXPERIMENT', 'n_folds', fallback = 10),
                                                    fewshot = config.getboolean('EXPERIMENT', 'few_shot', fallback = False),
                                                    verbose = config.getint('EXPERIMENT', 'verbosity', fallback = 1))
        
        print('Best parameters for regression (RMSE: {:.2f}): {!r}'.format(best_perf, best_params))
    
    else:
        
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
        datasets = dataset.data if isinstance(dataset, MultitaskRetrievalDataset) else [dataset]
        for di, dataset in enumerate(datasets):
            for lbl in query_classes:
                print('--- DATASET {}, CLASS {} ---'.format(di + 1, lbl))
                relevance, _ = dataset.class_relevance[lbl]
                lbl_best, lbl_perf = optimize_gp_params(dataset, relevance,
                                                        n_folds = config.getint('EXPERIMENT', 'n_folds', fallback = 10),
                                                        fewshot = config.getboolean('EXPERIMENT', 'few_shot', fallback = False),
                                                        verbose = config.getint('EXPERIMENT', 'verbosity', fallback = 1))
                best_params[(di,lbl)] = lbl_best
                best_perf[(di,lbl)] = lbl_perf
                print()

        # Print results
        for di, lbl in best_params.keys():
            print('Best parameters for dataset {}, class {} (AP: {:.2f}): {!r}'.format(di + 1, lbl, best_perf[(di,lbl)], best_params[(di,lbl)]))
