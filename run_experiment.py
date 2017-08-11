import sys, math

import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm, trange

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


def ndcg(y_true, y_score):
    
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


def plot_learning_step(dataset, query, relevance, learner, ret, fb):
    
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
        utils.plot_data(dataset.X_train, relevance, dataset.X_train[query], ret, axes[0,0])
        utils.plot_data(dataset.X_train, relevance, dataset.X_train[query], [r for i, r in enumerate(ret) if fb[i] != 0], axes[0,1])
        utils.plot_distribution(dataset.X_train, learner.rel_mean, dataset.X_train[query], axes[1,0])
        utils.plot_data(dataset.X_train, relevance, dataset.X_train[query], np.argsort(learner.rel_mean)[::-1][:np.sum(relevance > 0)], axes[1,1])
        fig.tight_layout()
        plt.show()
    
    else:
    
        raise RuntimeError("Don't know how to plot this dataset.")



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
    config, dataset, learner = utils.load_config(sys.argv[1], 'EXPERIMENT', overrides)
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
    aps, ndcgs = {}, {}
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
            it_ndcgs.append(ndcg(test_relevance, test_scores))
            
            for r in trange(config.getint('EXPERIMENT', 'rounds', fallback = 10), desc = 'Feedback rounds', leave = False, dynamic_ncols = True):
                
                ret = learner.fetch_unlabelled(config.getint('EXPERIMENT', 'batch_size'))
                fb = [0]
                while all(fbi == 0 for fbi in fb):
                    fb = simulate_feedback(relevance, ret, label_prob = config.getfloat('EXPERIMENT', 'label_prob', fallback = 1.0), mistake_prob = config.getfloat('EXPERIMENT', 'mistake_prob', fallback = 0.0))
                learner.update(dict(zip(ret, fb)))
                
                test_scores = learner.gp.predict(dataset.X_test_norm)
                it_aps.append(average_precision_score(test_relevance, test_scores))
                it_ndcgs.append(ndcg(test_relevance, test_scores))
                
                if plot:
                    plot_learning_step(dataset, query, relevance, learner, ret, fb)
            
            aps[lbl].append(it_aps)
            ndcgs[lbl].append(it_ndcgs)
    
    # Print mean and standard deviation for all iterations
    aps_mat = np.concatenate(list(aps.values()))
    ndcgs_mat = np.concatenate(list(ndcgs.values()))
    median_ap = np.median(aps_mat, axis = 0)
    median_ndcg = np.median(ndcgs_mat, axis = 0)
    mean_ap = aps_mat.mean(axis = 0)
    mean_ndcg = ndcgs_mat.mean(axis = 0)
    sd_ap = aps_mat.std(axis = 0)
    sd_ndcg = ndcgs_mat.std(axis = 0)
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
