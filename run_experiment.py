import sys

import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm, trange

from collections import OrderedDict

import utils



def simulate_feedback(labels, ret, label_prob = 0.8, mistake_prob = 0.05):
    
    fb = []
    for i in ret:
        if np.random.rand() >= label_prob:
            fb.append(0)
        elif (labels[i] > 0) and (np.random.rand() >= mistake_prob):
            fb.append(1)
        else:
            fb.append(-1)
    return fb



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
    
    # Read configuration and initialize environment
    config, dataset, learner = utils.load_config(config_file, 'EXPERIMENT', overrides)
    query_classes = str(config.get('EXPERIMENT', 'query_classes', fallback = '')).split()
    if len(query_classes) == 0:
        query_classes = list(dataset.class_relevance.keys())
    else:
        for i in range(len(query_classes)):
            try:
                query_classes[i] = int(query_classes[i])
            except ValueError:
                pass
    if plot or plot_hist:
        import matplotlib.pyplot as plt
    
    # Run multiple active retrieval rounds for each class
    aps, ndcgs = OrderedDict(), OrderedDict()
    for lbl in tqdm(query_classes, desc = 'Classes', leave = False, dynamic_ncols = True):
        relevance, test_relevance = dataset.class_relevance[lbl]
        aps[lbl] = []
        ndcgs[lbl] = []
        np.random.seed(0)
        queries = np.random.choice(np.nonzero(np.asarray(relevance) > 0)[0], config.getint('EXPERIMENT', 'repetitions', fallback = 10), replace = False)
        for query in tqdm(queries, desc = 'Class {}'.format(lbl), leave = False, dynamic_ncols = True):
            
            learner.reset()
            learner.update({ query : 1 })
            
            it_aps, it_ndcgs = [], []
            test_scores = learner.gp.predict(dataset.X_test_norm)
            it_aps.append(average_precision_score(test_relevance, test_scores))
            it_ndcgs.append(utils.ndcg(test_relevance, test_scores))
            
            for r in trange(config.getint('EXPERIMENT', 'rounds', fallback = 10), desc = 'Feedback rounds', leave = False, dynamic_ncols = True):
                
                ret = learner.fetch_unlabelled(config.getint('EXPERIMENT', 'batch_size'))
                fb = [0]
                while all(fbi == 0 for fbi in fb):
                    fb = simulate_feedback(relevance, ret, label_prob = config.getfloat('EXPERIMENT', 'label_prob', fallback = 1.0), mistake_prob = config.getfloat('EXPERIMENT', 'mistake_prob', fallback = 0.0))
                learner.update(dict(zip(ret, fb)))
                
                test_scores = learner.gp.predict(dataset.X_test_norm)
                it_aps.append(average_precision_score(test_relevance, test_scores))
                it_ndcgs.append(utils.ndcg(test_relevance, test_scores))
                
                if plot:
                    utils.plot_learning_step(dataset, query, relevance, learner, ret, fb)
            
            aps[lbl].append(it_aps)
            ndcgs[lbl].append(it_ndcgs)
    
    # Print mean and standard deviation for all iterations
    if config.get('EXPERIMENT', 'avg_class_perf', fallback = True):
        aps_mat = OrderedDict([('Overall Performance', np.concatenate(list(aps.values())))])
        ndcgs_mat = OrderedDict([('Overall Performance', np.concatenate(list(ndcgs.values())))])
    else:
        aps_mat = aps
        ndcgs_mat = ndcgs
    for lbl in aps_mat.keys():
        if len(aps_mat) > 1:
            print('\n{}\n{:-<{}}\n'.format(lbl if isinstance(lbl, str) else 'Class {}'.format(lbl), '', len(lbl) if isinstance(lbl, str) else len(str(lbl)) + 6))
        median_ap = np.median(aps_mat[lbl], axis = 0)
        median_ndcg = np.median(ndcgs_mat[lbl], axis = 0)
        mean_ap = np.mean(aps_mat[lbl], axis = 0)
        mean_ndcg = np.mean(ndcgs_mat[lbl], axis = 0)
        sd_ap = np.std(aps_mat[lbl], axis = 0)
        sd_ndcg = np.std(ndcgs_mat[lbl], axis = 0)
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
        for i, lbl in enumerate(aps.keys()):
            axes[i,0].set_ylabel('Class {}'.format(lbl))
            for r in range(rounds):
                axes[i,r].hist(np.array(aps[lbl])[:,r])
        fig.tight_layout()
        plt.show()
