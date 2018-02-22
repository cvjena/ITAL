import sys

import numpy as np
import scipy.stats
from sklearn.metrics import average_precision_score, mean_squared_error
from tqdm import tqdm, trange

from collections import OrderedDict

import utils, viz_utils
from datasets import MultitaskRetrievalDataset
from italia.regression_base import ActiveRegressionBase



def simulate_retrieval_feedback(labels, ret, label_prob = 0.8, mistake_prob = 0.05):
    
    fb = []
    for i in ret:
        if np.random.rand() >= label_prob:
            fb.append(0)
        elif np.random.rand() >= mistake_prob:
            fb.append(labels[i])
        elif labels[i] == 0:
            fb.append(np.random.choice([-1, 1]))
        elif labels[i] > 0:
            fb.append(-1)
        else:
            fb.append(1)
    return fb


def simulate_regression_feedback(y_true, ret, label_prob = 0.8, mistake_variance = 0.0):
    
    fb = []
    for i in ret:
        if np.random.rand() >= label_prob:
            fb.append(None)
        elif mistake_variance <= 0:
            fb.append(y_true[i])
        else:
            fb.append(scipy.stats.norm.rvs(y_true[i], mistake_variance))
    return fb



def run_retrieval_experiment(config, dataset, learner, plot = False, plot_hist = False):
    
    if isinstance(dataset, MultitaskRetrievalDataset):
        multitask = True
        num_datasets = len(dataset)
        datasets = dataset.datasets()
    else:
        multitask = False
        num_datasets = 1
        datasets = [dataset]
    aps, ndcgs = OrderedDict(), OrderedDict()
    
    dataset_iterator = enumerate(datasets)
    if num_datasets > 1:
        dataset_iterator = tqdm(dataset_iterator, desc = 'Datasets', total = num_datasets, leave = False, dynamic_ncols = True)
    for di, dataset in dataset_iterator:
        
        if multitask:
            learner.fit(dataset.X_train_norm)
        
        # Get classes to draw queries from
        query_classes = str(config.get('EXPERIMENT', 'query_classes', fallback = '')).split()
        if len(query_classes) == 0:
            query_classes = list(dataset.class_relevance.keys())
        else:
            for i in range(len(query_classes)):
                try:
                    query_classes[i] = int(query_classes[i])
                except ValueError:
                    pass

        # Run multiple active retrieval rounds for each class
        num_initial_negatives = config.getint('EXPERIMENT', 'initial_negatives', fallback = 0)
        for lbl in tqdm(query_classes, desc = 'Classes', leave = False, dynamic_ncols = True):
            relevance, test_relevance = dataset.class_relevance[lbl]
            nonzero_test_relevance = (np.asarray(test_relevance) != 0)
            aps[(di,lbl)] = []
            ndcgs[(di,lbl)] = []
            np.random.seed(0)
            queries = np.random.choice(np.nonzero(np.asarray(relevance) > 0)[0], (config.getint('EXPERIMENT', 'repetitions', fallback = 10), config.getint('EXPERIMENT', 'num_init', fallback = 1)), replace = False)
            for query in tqdm(queries, desc = 'Class {}'.format(lbl), leave = False, dynamic_ncols = True):

                learner.reset()
                learner.update({ q : 1 for q in query })

                if num_initial_negatives > 0:
                    neg_ind = np.argpartition(learner.rel_mean, num_initial_negatives - 1)[:num_initial_negatives]
                    learner.update({ ni : -1 for ni in neg_ind })

                it_aps, it_ndcgs = [], []
                test_scores = learner.gp.predict(dataset.X_test_norm)
                it_aps.append(average_precision_score(np.asarray(test_relevance)[nonzero_test_relevance], np.asarray(test_scores)[nonzero_test_relevance]))
                it_ndcgs.append(utils.ndcg(test_relevance, test_scores))

                if plot:
                    viz_utils.plot_learning_step(dataset, query, relevance, learner, [], [])

                for r in trange(config.getint('EXPERIMENT', 'rounds', fallback = 10), desc = 'Feedback rounds', leave = False, dynamic_ncols = True):

                    ret = learner.fetch_unlabelled(config.getint('EXPERIMENT', 'batch_size'))
                    fb = simulate_retrieval_feedback(relevance, ret, label_prob = config.getfloat('EXPERIMENT', 'label_prob', fallback = 1.0), mistake_prob = config.getfloat('EXPERIMENT', 'mistake_prob', fallback = 0.0))
                    learner.update(dict(zip(ret, fb)))

                    test_scores = learner.gp.predict(dataset.X_test_norm)
                    it_aps.append(average_precision_score(np.asarray(test_relevance)[nonzero_test_relevance], np.asarray(test_scores)[nonzero_test_relevance]))
                    it_ndcgs.append(utils.ndcg(test_relevance, test_scores))

                    if plot:
                        viz_utils.plot_learning_step(dataset, query, relevance, learner, ret, fb)

                aps[(di,lbl)].append(it_aps)
                ndcgs[(di,lbl)].append(it_ndcgs)
    
    # Print mean and standard deviation for all iterations
    if config.get('EXPERIMENT', 'avg_class_perf', fallback = True):
        aps_mat = OrderedDict([((-1, 'Overall Performance'), np.concatenate(list(aps.values())))])
        ndcgs_mat = OrderedDict([((-1, 'Overall Performance'), np.concatenate(list(ndcgs.values())))])
    else:
        aps_mat = aps
        ndcgs_mat = ndcgs
    for di, lbl in aps_mat.keys():
        if len(aps_mat) > 1:
            title = '{}{}'.format('Dataset {}, '.format(di+1) if (num_datasets > 1) and (di >= 0) else '', lbl if isinstance(lbl, str) else 'Class {}'.format(lbl))
            print('\n{}\n{:-<{}}\n'.format(title, '', len(title)))
        median_ap = np.median(aps_mat[(di,lbl)], axis = 0)
        median_ndcg = np.median(ndcgs_mat[(di,lbl)], axis = 0)
        mean_ap = np.mean(aps_mat[(di,lbl)], axis = 0)
        mean_ndcg = np.mean(ndcgs_mat[(di,lbl)], axis = 0)
        sd_ap = np.std(aps_mat[(di,lbl)], axis = 0)
        sd_ndcg = np.std(ndcgs_mat[(di,lbl)], axis = 0)
        print('Round;Median_AP;Mean_AP;AP_SD;Median_NDCG;Mean_NDCG;NDCG_SD')
        for i in range(len(mean_ap)):
            print('{};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f}'.format(i, median_ap[i], mean_ap[i], sd_ap[i], median_ndcg[i], mean_ndcg[i], sd_ndcg[i]))
    
    # Plot AP histogram
    if plot_hist:
        rounds = config.getint('EXPERIMENT', 'rounds', fallback = 10) + 1
        fig, axes = plt.subplots(len(aps), rounds, figsize = (2.5 * rounds, 2 * len(aps)))
        if len(aps) == 1:
            axes = axes[None,:]
        elif rounds == 1:
            axes = axes[:,None]
        for r in range(rounds):
            axes[0,r].set_title('Round {}'.format(r))
        for i, (di,lbl) in enumerate(aps.keys()):
            axes[i,0].set_ylabel('{}Class {}'.format('Dataset {}, '.format(di+1) if (num_datasets > 1) and (di >= 0) else '', lbl))
            for r in range(rounds):
                axes[i,r].hist(np.array(aps[(di,lbl)])[:,r])
        fig.tight_layout()
        plt.show()



def run_regression_experiment(config, dataset, learner, plot = False, plot_hist = False):
    
    # Run multiple rounds of active retrieval with different initializations
    rmses = []
    np.random.seed(0)
    initializations = [np.random.choice(len(dataset.X_train), config.getint('EXPERIMENT', 'num_init', fallback = 10), replace = False) for i in range(config.getint('EXPERIMENT', 'repetitions', fallback = 10))]
    for init in tqdm(initializations, desc = 'Repetitions', leave = False, dynamic_ncols = True):
        
        learner.reset()
        learner.update({ i : dataset.y_train[i] for i in init })
        
        test_predictions = learner.gp.predict(dataset.X_test_norm)
        it_rmses = [np.sqrt(mean_squared_error(dataset.y_test, test_predictions))]
        
        if plot:
            viz_utils.plot_regression_step(dataset, init, learner, [], [])
        
        for r in trange(config.getint('EXPERIMENT', 'rounds', fallback = 10), desc = 'Feedback rounds', leave = False, dynamic_ncols = True):
            
            ret = learner.fetch_unlabelled(config.getint('EXPERIMENT', 'batch_size'))
            fb = [None]
            while all(fbi is None for fbi in fb):
                fb = simulate_regression_feedback(dataset.y_train, ret, label_prob = config.getfloat('EXPERIMENT', 'label_prob', fallback = 1.0), mistake_variance = config.getfloat('EXPERIMENT', 'mistake_variance', fallback = 0.0))
            learner.update(dict(zip(ret, fb)))
            
            test_predictions = learner.gp.predict(dataset.X_test_norm)
            it_rmses.append(np.sqrt(mean_squared_error(dataset.y_test, test_predictions)))
            
            if plot:
                viz_utils.plot_regression_step(dataset, init, learner, ret, fb)
        
        rmses.append(it_rmses)
    
    # Print mean and standard deviation for all iterations
    rmses = np.array(rmses)
    median_rmse = np.median(rmses, axis = 0)
    mean_rmse = np.mean(rmses, axis = 0)
    sd_rmse = np.std(rmses, axis = 0)
    print('Round;Median_RMSE;Mean_RMSE;RMSE_SD')
    for i in range(len(mean_rmse)):
        print('{};{:.4f};{:.4f};{:.4f}'.format(i, median_rmse[i], mean_rmse[i], sd_rmse[i]))
    
    # Plot RMSE histogram
    if plot_hist:
        fig, axes = plt.subplots(1, rmses.shape[1], figsize = (2.5 * rmses.shape[1], 3))
        for r in range(rmses.shape[1]):
            axes[r].set_title('Round {}'.format(r))
            axes[r].hist(rmses[:,r])
        fig.tight_layout()
        plt.show()



if __name__ == '__main__':
    
    # Parse arguments
    config_file = None
    overrides = {}
    plot = False
    plot_hist = False
    for arg in sys.argv[1:]:
        if arg.lower() == '--plot':
            plot = True
        elif arg.lower() == '--hist':
            plot_hist = True
        elif arg.startswith('--'):
            k, v = arg[2:].split('=', maxsplit = 1)
            overrides[k] = v
        elif config_file is None:
            config_file = arg
        else:
            print('Unexpected argument: {}'.format(arg))
            exit()
    if config_file is None:
        print('Usage: {} <experiment-config-file> [--plot] [--hist] [--<override-option>=<override-value> ...]'.format(sys.argv[0]))
        exit()
    
    if plot or plot_hist:
        import matplotlib.pyplot as plt
    
    # Read configuration and initialize environment
    config, dataset, learner = utils.load_config(config_file, 'EXPERIMENT', overrides)
    if isinstance(learner, ActiveRegressionBase):
        run_regression_experiment(config, dataset, learner, plot = plot, plot_hist = plot_hist)
    else:
        run_retrieval_experiment(config, dataset, learner, plot = plot, plot_hist = plot_hist)
