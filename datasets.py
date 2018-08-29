import numpy as np
import csv
import os.path
import pickle
from glob import glob

import sklearn.datasets
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal as mvn
import scipy.io



#################
##  Utilities  ##
#################


def load_dataset(dataset_name, **kwargs):
    """ Instantiates a dataset by its name.
    
    # Arguments:
    - dataset_name: The name of the dataset. Appending 'Dataset' to it should result in the name of the class to be intantiated.
    
    Additional arguments will be passed through to the constructor of the dataset.
    
    # Returns:
        object
    """
    
    class_name = dataset_name + 'Dataset'
    if class_name not in globals():
        raise ValueError('Unknown dataset: {}'.format(dataset_name))
    return globals()[class_name](**kwargs)



####################
##  Base Classes  ##
####################


class Dataset(object):
    """ A dataset.
    
    # Properties:
    
    - X: array of all samples.
    
    - y: array with labels of all samples.
    
    - X_train: array with training data.
    
    - y_train: array with training labels.
    
    - X_test: array with validation data.
    
    - y_test: array with validation labels.
    
    - X_max: maximum value in X_train.
    
    - X_min: minimum value in X_train.
    
    - X_train_norm: X_train scaled to [0,1].
    
    - X_test_norm: X_test scaled by the same parameters used to obtain X_train_norm.
    """
    
    def __init__(self, X, y, X_test = None, y_test = None, test_size = 0.2):
        """ Initializes a dataset from given data, optionally automatically splitting it into training and validation set.
        
        # Arguments:
        
        - X: array of all samples in case of automatic splitting, otherwise array of training samples.
        
        - y: array with labels of all samples in case of automatic splitting, otherwise array of training labels
        
        - X_test: optionally, array of test samples.
        
        - y_test: optionally, array of test labels.
        
        - test_size: either a float specifying the fraction of the data to be used for
                     validation or an integer specifying the absolute number of
                     validation samples. Has no effect if X_test and y_test are given explicitly.
        """
        
        if (X_test is None) or (y_test is None):
        
            self.X = np.array(X)
            self.y = np.array(y)

            self.X_train, self.X_test, self.y_train, self.y_test, self.ind_train, self.ind_test = train_test_split(self.X, self.y, np.arange(len(self.X)), test_size = test_size, random_state = 0)
        
        else:
            
            self.X_train = np.array(X)
            self.y_train = np.array(y)
            self.X_test = np.array(X_test)
            self.y_test = np.array(y_test)
            
            self.X = np.concatenate([self.X_train, self.X_test])
            self.y = np.concatenate([self.y_train, self.y_test])
        
        self._preprocess()
    
    
    def _preprocess(self):
        """ Computes X_max, X_min, X_train_norm, X_test_norm, labels, and class_relevance. """
        
        self.X_max, self.X_min = self.X_train.max(), self.X_train.min()
        self.X_train_norm = (self.X_train - self.X_min) / (self.X_max - self.X_min)
        self.X_test_norm = (self.X_test - self.X_min) / (self.X_max - self.X_min)



class RetrievalDataset(Dataset):
    """ A dataset for information retrieval.
    
    # Properties (in addition to those inherited from Dataset):
    
    - labels: list of unique labels.
    
    - class_relevance: dictionary mapping labels to arrays specifying whether a sample
                       is relevant for that label. Class relevance is given as 1, -1, or 0
                       if it is not certain whether the label belongs to the class or not.
    
    - queries: pre-defined queries to be used instead of random ones, given as indices dictionary
               mapping labels to lists of lists of indices in the training dataset. For each label,
               there should be a list containing queries. Each query is a list of image indices
               being used as initial training images for that query.
    """
    
    def __init__(self, *args, **kwargs):
        
        Dataset.__init__(self, *args, **kwargs)
        self.imgs = self.imgs_train = self.imgs_test = None
        if 'queries' not in self.__dict__:
            self.queries = None
    
    
    def _preprocess(self):
        
        Dataset._preprocess(self)
        
        self.labels = np.unique(self.y)
        self.class_relevance = { lbl : (2 * (self.y_train == lbl) - 1, 2 * (self.y_test == lbl) - 1) for lbl in self.labels }



class MultilabelRetrievalDataset(RetrievalDataset):
    """ A dataset for information retrieval where samples can belong to multiple labels.
    
    For such datasets, the properties y, y_train, and y_test are lists of lists of labels.
    """
    
    def __init__(self, X, y, X_test = None, y_test = None, imgs = None, imgs_test = None, test_size = 0.2):
        """ Initializes a dataset from given data, optionally automatically splitting it into training and validation set.
        
        # Arguments:
        
        - X: array of all samples in case of automatic splitting, otherwise array of training samples.
        
        - y: list with lists of labels for all samples in case of automatic splitting, otherwise array of training labels
        
        - X_test: optionally, array of test samples.
        
        - y_test: optionally, list of test labels.
        
        - imgs: optionally, list of images belonging to the features in X.
        
        - imgs_test: optionally, list of images belonging to the features in X_test.
        
        - test_size: either a float specifying the fraction of the data to be used for
                     validation or an integer specifying the absolute number of
                     validation samples. Has no effect if X_test and y_test are given explicitly.
        """
        
        if (X_test is None) or (y_test is None):
        
            self.X = np.array(X)
            self.y = list(y)

            if imgs:
                self.imgs = imgs
                self.X_train, self.X_test, self.y_train, self.y_test, self.ind_train, self.ind_test, self.imgs_train, self.imgs_test = train_test_split(self.X, self.y, np.arange(len(self.X)), self.imgs, test_size = test_size, random_state = 0)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test, self.ind_train, self.ind_test = train_test_split(self.X, self.y, np.arange(len(self.X)), test_size = test_size, random_state = 0)
                self.imgs = self.imgs_train = self.imgs_test = None
        
        else:
            
            self.X_train = np.array(X)
            self.y_train = list(y)
            self.X_test = np.array(X_test)
            self.y_test = list(y_test)
            self.imgs_train = imgs if imgs else None
            self.imgs_test = imgs_test if imgs_test else None
            
            self.X = np.concatenate([self.X_train, self.X_test])
            self.y = self.y_train + self.y_test
            if (self.imgs_train is not None) and (self.imgs_test is not None):
                self.imgs = self.imgs_train + self.imgs_test
            else:
                self.imgs = None
        
        self._preprocess()
    
    
    def _preprocess(self):
        
        Dataset._preprocess(self)
        
        self.labels = set(lbl for lbls in self.y for lbl in lbls)
        self.class_relevance = { lbl : (
            np.array([1 if lbl in lbls else -1 for lbls in self.y_train]),
            np.array([1 if lbl in lbls else -1 for lbls in self.y_test])
        ) for lbl in self.labels }



class MultitaskRetrievalDataset(object):
    """ A collection of several similar retrieval datasets. Each sub-dataset constitutes a binary classification task.
    
    Use the datasets() method to obtain an iterator over the individual RetrievalDataset instances.
    """
    
    def __len__(self):
        return 0
    
    def datasets(self):
        raise NotImplementedError()



class RegressionDataset(Dataset):
    """ A dataset for regression. """
    
    pass



##########################
##  Retrieval Datasets  ##
##########################


class StoredDataset(RetrievalDataset):
    """ Loads a dataset from a numpy file. """
    
    def __init__(self, data_file, **kwargs):
        """ Loads the dataset.
        
        # Arguments:
        - data_file: path to a .npz file containing X_train, y_train, X_test, and y_test.
        """
        
        data = np.load(data_file)
        self.X_train, self.y_train = data['X_train'], data['y_train']
        self.X_test, self.y_test = data['X_test'], data['y_test']
        
        self.X = np.concatenate((self.X_train, self.X_test))
        self.y = np.concatenate((self.y_train, self.y_test))
        
        self.imgs = self.imgs_train = self.imgs_test = None
        
        self._preprocess()



class ToyDataset(RetrievalDataset):
    """ Generates a toy dataset. """
    
    def __init__(self, size_factor = 10, test_size = 0.5, **kwargs):
        """ Initializes the dataset.
        
        # Arguments:
        - size_factor: controls to size of the dataset, which will be 17 * size_factor.
        """
        
        np.random.seed(0)
        X = np.concatenate([
                mvn.rvs([20.0, 30.0], np.random.randn(2,2) + np.eye(2) * 2, 2 * size_factor),
                mvn.rvs([10.0, 10.0], np.random.randn(2,2) + np.eye(2) * 2, 3 * size_factor),
                mvn.rvs([30.0, 15.0], np.random.randn(2,2) + np.eye(2) * 2, 6 * size_factor),
                mvn.rvs([15.0, 20.0], np.random.randn(2,2) + np.eye(2) * 2, 4 * size_factor),
                mvn.rvs([25.0, 30.0], np.random.randn(2,2) + np.eye(2) * 2, 2 * size_factor)
            ])
        y = np.concatenate([np.ones(5 * size_factor, dtype=int), np.zeros(12 * size_factor, dtype=int)])
        
        RetrievalDataset.__init__(self, X, y, test_size=test_size, **kwargs)



class IrisDataset(RetrievalDataset):
    """ Interface to the Iris dataset. """
    
    def __init__(self, **kwargs):
        
        X, y = sklearn.datasets.load_iris(return_X_y = True)
        RetrievalDataset.__init__(self, X, y, **kwargs)



class WineDataset(RetrievalDataset):
    """ Interface to the UCI Wine dataset.
    
    https://archive.ics.uci.edu/ml/datasets/wine
    """
    
    def __init__(self, data_file, **kwargs):
        """ Loads the Wine dataset.
        
        # Arguments:
        - data_file: path to wine.data.
        """
        
        X = np.loadtxt(data_file, delimiter = ',', dtype = float)[:,1:]
        y = np.loadtxt(data_file, delimiter = ',', dtype = int, usecols = 0)
        RetrievalDataset.__init__(self, X, y, **kwargs)



class LeafDataset(RetrievalDataset):
    """ Interface to the UCI Leaf dataset.
    
    https://archive.ics.uci.edu/ml/datasets/leaf
    """
    
    def __init__(self, data_file, test_size = 0.5, **kwargs):
        """ Loads the Wine dataset.
        
        # Arguments:
        - data_file: path to leaf.csv.
        """
        
        X = np.loadtxt(data_file, delimiter = ',', dtype = float)[:,2:]
        y = np.loadtxt(data_file, delimiter = ',', dtype = int, usecols = 0)
        RetrievalDataset.__init__(self, X, y, test_size=test_size, **kwargs)



class USPSDataset(RetrievalDataset):
    """ Interface to the USPS dataset.
    
    https://www-i6.informatik.rwth-aachen.de/~keysers/usps.html
    """
    
    def __init__(self, train_data_file, test_data_file, **kwargs):
        """ Loads the USPS dataset.
        
        # Arguments:
        - train_data_file: path to usps_train.jf.
        - test_data_file: path to usps_test.jf.
        """
        
        self.X_train, self.y_train = self._read_usps(train_data_file)
        self.X_test, self.y_test = self._read_usps(test_data_file)
        
        self.X = np.concatenate((self.X_train, self.X_test))
        self.y = np.concatenate((self.y_train, self.y_test))
        
        self.imgs = self.X.reshape(-1, 16, 16)
        self.imgs_train = self.X_train.reshape(-1, 16, 16)
        self.imgs_test = self.X_test.reshape(-1, 16, 16)
        
        self._preprocess()
    
    
    def _read_usps(self, data_file):
        
        with open(data_file) as f:
            f.readline()
            X, y = [], []
            for line in f:
                if (line.strip() == '') or (line.strip() == '-1'):
                    break
                data = line.strip().split()
                y.append(int(data[0]))
                X.append([float(x) for x in data[1:]])
        
        return np.array(X), np.array(y)



class NaturalScenesDataset(RetrievalDataset):
    """ Interface to the 13 Natural Scenes dataset.
    
    http://vision.stanford.edu/resources_links.html
    """
    
    def __init__(self, pickle_dump, **kwargs):
        """ Loads the Natural Scenes dataset.
        
        # Arguments:
        - pickle_dump: Path to a pickle dump containing the items 'X_pca' and 'y'.
        """
        
        with open(pickle_dump, 'rb') as f:
            dump = pickle.load(f)
        
        RetrievalDataset.__init__(self, dump['X_pca'], dump['y'], **kwargs)



class OxfordDataset(RetrievalDataset):
    """ Interface to the Oxford5k dataset.
    
    http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
    """
    
    def __init__(self, feat_dump, gt_dir, img_dir = None, **kwargs):
        """ Loads the Oxford dataset.
        
        # Arguments:
        
        - feat_dump: Path to a pickle file containing a dictionary with an item 'feat' that refers to a
                     a dictionary mapping image IDs to feature vectors.
        
        - gt_dir: Path to a directory containing ground-truth label and query files.
                  For each label, there must be files suffixed with '_good.txt', '_ok.txt', and '_junk.txt'
                  listing one image ID per line.
                  In addition, there must be a file suffixed '_query.txt' for each label, containing lines
                  with query image IDs and bounding box coordinates (which are just ignored by this dataset interface).
        
        - img_dir: Path to a directory containing the Oxford5k images.
        """
        
        with open(feat_dump, 'rb') as feat_file:
            dump = pickle.load(feat_file)['feat']
            self.ids = list(dump.keys())
            self.id2ind = { id : i for i, id in enumerate(self.ids) }
            X = np.stack(list(dump.values()))
            del dump
        
        y = [-1] * len(self.ids)
        self.is_junk = [False] * len(self.ids)
        self.queries = {}
        
        gt_files = glob(os.path.join(gt_dir, '*_good.txt'))
        for lbl, gt_file in enumerate(gt_files):
            with open(gt_file) as f:
                for l in f:
                    if l.strip() != '':
                        y[self.id2ind[l.strip()]] = lbl
            with open(gt_file.replace('_good.txt', '_ok.txt')) as f:
                for l in f:
                    if l.strip() != '':
                        y[self.id2ind[l.strip()]] = lbl
            with open(gt_file.replace('_good.txt', '_junk.txt')) as f:
                for l in f:
                    if l.strip() != '':
                        y[self.id2ind[l.strip()]] = lbl
                        self.is_junk[self.id2ind[l.strip()]] = True
            self.queries[lbl] = []
            with open(gt_file.replace('_good.txt', '_query.txt')) as f:
                for l in f:
                    if l.strip() != '':
                        self.queries[lbl].append([self.id2ind[l.strip().split()[0]]])
                        y[self.id2ind[l.strip().split()[0]]] = lbl
        
        y = np.asarray(y)
        test_ind = np.setdiff1d(np.arange(X.shape[0]), [qind[0] for q in self.queries.values() for qind in q])
        super(OxfordDataset, self).__init__(X, y, X[test_ind], y[test_ind], **kwargs)

        if img_dir is not None:
            self.imgs = self.imgs_train = [os.path.join(img_dir, '{}.jpg'.format(id)) for id in self.ids]
            self.imgs_test = np.asarray(self.imgs)[test_ind].tolist()
    
    
    def _preprocess(self):
        
        Dataset._preprocess(self)
        
        self.labels = np.unique(self.y)
        self.class_relevance = { lbl : (
                np.where(self.is_junk, np.zeros(len(self.y_train)), 2 * (self.y_train == lbl) - 1),
                np.where(np.asarray(self.is_junk)[np.setdiff1d(np.arange(len(self.is_junk)), [qind[0] for q in self.queries.values() for qind in q])], np.zeros(len(self.y_test)), 2 * (self.y_test == lbl) - 1)
            ) for lbl in self.labels }



class MIRFLICKRDataset(MultilabelRetrievalDataset):
    """ Interface to the MIRFLICKR dataset.
    
    http://press.liacs.nl/mirflickr/
    """
    
    def __init__(self, feat_dump, gt_dir, gt_pattern = '*_r1.txt', img_dir = None, **kwargs):
        """ Loads the MIRFLICKR dataset.
        
        # Arguments:
        
        - feat_dump: Path to a numpy file containing a 2-D array with features.
        
        - gt_dir: Path to a directory containing ground-truth label files, one file per label.
                  The content of each file must be a list of image IDs (counting from 1) belonging
                  to the class, one ID per line.
        
        - gt_pattern: Glob pattern for the ground-truth label files in the given directory.
        
        - img_dir: Path to a directory containing the MIRFLICKR images.
        """
        
        X = np.load(feat_dump)
        y = [[] for i in range(X.shape[0])]
        self.y_wide = [[] for i in range(X.shape[0])]
        
        gt_files = glob(os.path.join(gt_dir, gt_pattern))
        for lbl, gt_file in enumerate(gt_files):
            with open(gt_file) as f:
                for l in f:
                    if l.strip() != '':
                        y[int(l.strip()) - 1].append(lbl)
            with open(gt_file[:-7] + '.txt') as f:
                for l in f:
                    if l.strip() != '':
                        self.y_wide[int(l.strip()) - 1].append(lbl)
        
        if img_dir is not None:
            imgs = [os.path.join(img_dir, 'im{}.jpg'.format(i+1)) for i in range(X.shape[0])]
        else:
            imgs = None
        
        MultilabelRetrievalDataset.__init__(self, X, y, imgs = imgs, **kwargs)
    
    
    def _preprocess(self):
        
        Dataset._preprocess(self)
        
        self.y_wide_train = [self.y_wide[ind] for ind in self.ind_train]
        self.y_wide_test = [self.y_wide[ind] for ind in self.ind_test]
        
        self.labels = set(lbl for lbls in self.y for lbl in lbls)
        self.class_relevance = { lbl : (
            np.array([1 if lbl in lbls else (0 if lbl in wide_labels else -1) for lbls, wide_labels in zip(self.y_train, self.y_wide_train)]),
            np.array([1 if lbl in lbls else (0 if lbl in wide_labels else -1) for lbls, wide_labels in zip(self.y_test, self.y_wide_test)])
        ) for lbl in self.labels }



class ImageNetDataset(MultitaskRetrievalDataset):
    """ Interface to ILSVRC 2010 dataset.
    
    This dataset consists of several random binary classification tasks.
    A single positive and multiple negative classes are selected for each task.
    """
    
    def __init__(self, sbow_dir, meta_file, val_label_file, train_img_dir = None, val_img_dir = None, num_tasks = 100, num_negative_classes = 9, task_file = None):
        """ Initializes the ImageNet dataset.
        
        # Arguments:
        
        - sbow_dir: directory containing the sub-directories "train" and "val" with
                    pre-computed Bag-of-Words features.
        
        - meta_file: path to meta.mat file of the ILSVRC 2010 development kit.
        
        - val_label_file: path to ILSVRC2010_validation_ground_truth.txt file of the ILSVRC 2010 development kit.
        
        - train_img_dir: optionally, path to the directory containing the tar archives with images of the training synsets.
                         Only needed for filling the `imgs_train` attributes of the datasets.
        
        - val_img_dir: optionally, path to the directory containing the validation images.
                       Only needed for filling the `imgs_train` attributes of the datasets.
        
        - num_tasks: number of random binary classification tasks.
        
        - num_negative_classes: number of negative classes per task.
        
        - task_file: Optionally, path to a text file specifying the retrieval tasks, one per line as white-space-separated list of synsets.
                     The first synset in each list is the positive class and the remaining ones are negative.
                     If this file exists, the specified tasks will override `num_tasks` and `num_negative`.
                     If a path to a non-existent file is specified, the tasks generated according to `num_tasks` and `num_negative`
                     will be stored in that file.
        """
        
        MultitaskRetrievalDataset.__init__(self)
        
        self.sbow_dir = sbow_dir
        self.meta_file = meta_file
        self.val_label_file = val_label_file
        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir
        self.num_tasks = num_tasks
        self.num_negative_classes = num_negative_classes
        
        # Load metadata
        self._load_meta(meta_file, val_label_file)
        self._load_val_data()
        
        # Create random subsets
        np.random.seed(0)
        if (task_file is not None) and os.path.exists(task_file):
            with open(task_file) as f:
                self.selected_synsets = [[self.synsets.index(syn) for syn in l.strip().split()] for l in f if l.strip() != '']
        else:
            self.selected_synsets = [np.random.choice(len(self.synsets), num_negative_classes + 1, replace = False) for i in range(num_tasks)]
            if task_file is not None:
                with open(task_file, 'w') as f:
                    f.write('\n'.join(' '.join(self.synsets[syn] for syn in synsets) for synsets in self.selected_synsets))
    
    
    def __len__(self):
        """ Returns the number of random binary classification tasks. """
        
        return len(self.selected_synsets)
    
    
    def datasets(self):
        """ Generator over all random binary classification tasks as RetrievalDataset instances. """
        
        for synsets in self.selected_synsets:
            pos_synset = synsets[0]
            neg_synsets = synsets[1:]
            X_train, y_train, imgs_train = self._load_train_data(pos_synset, neg_synsets)
            X_test = self.val_feat[self.val_labels == pos_synset]
            y_test = np.ones(len(X_test))
            for syn in neg_synsets:
                X_test = np.concatenate([X_test, self.val_feat[self.val_labels == syn]])
            y_test = np.concatenate([y_test, np.zeros(len(X_test) - len(y_test))])
            
            ds = RetrievalDataset(X_train, y_train, X_test, y_test)
            if self.train_img_dir:
                ds.imgs_train = [(os.path.join(self.train_img_dir, '{}.tar'.format(img_id.split('_')[0])), '{}.JPEG'.format(img_id)) for img_id in imgs_train]
            if self.val_img_dir:
                ds.imgs_test = [os.path.join(self.val_img_dir, '{}.JPEG'.format(img_id)) for img_id, lbl in zip(self.val_ids, self.val_labels) if lbl in [pos_synset] + list(neg_synsets)]
            if (ds.imgs_train is not None) and (ds.imgs_test is not None):
                ds.imgs = ds.imgs_train + ds.imgs_test
            yield ds
    
    
    def _load_meta(self, meta_file, val_label_file):
        
        self.synsets = [syn[0] for syn in scipy.io.loadmat(meta_file)['synsets']['WNID'][:1000,0]]
        
        with open(val_label_file) as vf:
            self.val_labels = np.array([int(l.strip()) - 1 for l in vf if l.strip() != ''])
    
    
    def _load_train_data(self, pos_synset, neg_synsets):
        
        X, img_ids = self._load_feat(os.path.join(self.sbow_dir, 'train', '{}.sbow.mat'.format(self.synsets[pos_synset])))
        y = np.ones(len(X))
        
        for syn in neg_synsets:
            neg_feat, neg_ids = self._load_feat(os.path.join(self.sbow_dir, 'train', '{}.sbow.mat'.format(self.synsets[syn])))
            X = np.concatenate([X, neg_feat])
            img_ids += neg_ids
        y = np.concatenate([y, np.zeros(len(X) - len(y))])
        
        return X, y, img_ids
    
    
    def _load_val_data(self):
        
        feat, ids = [], []
        for i in range(1, 51):
            X, img_ids = self._load_feat(os.path.join(self.sbow_dir, 'val', 'val.{:04d}.sbow.mat'.format(i)))
            feat.append(X)
            ids += img_ids
        
        self.val_feat = np.concatenate(feat)
        self.val_ids = np.array(ids)
    
    
    def _load_feat(self, mat_file):
        
        feat = scipy.io.loadmat(mat_file)['image_sbow'].ravel()
        X = np.array([np.bincount(f[1]['word'][0,0].ravel(), minlength = 1000) for f in feat], dtype = float)
        X /= np.linalg.norm(X, axis = -1, keepdims = True)
        ids = [f[0][0] for f in feat]
        return X, ids



###########################
##  Regression Datasets  ##
###########################


class ToyRegressionDataset(RegressionDataset):
    """ Generates a toy dataset for regression. """
    
    def __init__(self, num_samples = 200, test_size = 0.5, **kwargs):
        """ Initializes the dataset.
        
        # Arguments:
        - num_samples: total number of samples in the dataset.
        """
        
        np.random.seed(0)
        X = np.random.uniform(0.0, 2 * np.pi, size = (num_samples, 2))
        y = (np.sin(X[:,0]) + np.sin(X[:,1])) * np.cos(X.sum(axis = -1))
        
        RegressionDataset.__init__(self, X, y, test_size=test_size, **kwargs)



class WinequalityDataset(RegressionDataset):
    """ Interface to the UCI Wine-quality dataset.
    
    https://archive.ics.uci.edu/ml/datasets/wine+quality
    """
    
    def __init__(self, data_file, **kwargs):
        """ Loads the Wine-quality dataset.
        
        # Arguments:
        - data_file: path to winequality-{red|white}.csv.
        """
        
        data = np.loadtxt(data_file, delimiter = ';', skiprows = 1, dtype = float)
        RegressionDataset.__init__(self, data[:,:-1], data[:,-1], **kwargs)



class ConcreteDataset(RegressionDataset):
    """ Interface to the UCI Concrete dataset.
    
    https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    """
    
    def __init__(self, data_file, **kwargs):
        """ Loads the Concrete dataset.
        
        # Arguments:
        - data_file: path to Concrete_Data.csv.
        """
        
        data = np.loadtxt(data_file, delimiter = ';', skiprows = 1, dtype = float)
        RegressionDataset.__init__(self, data[:,:-1], data[:,-1], **kwargs)



class YachtDataset(RegressionDataset):
    """ Interface to the UCI Yacht dataset.
    
    https://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
    """
    
    def __init__(self, data_file, **kwargs):
        """ Loads the Yacht dataset.
        
        # Arguments:
        - data_file: path to yacht_hydrodynamics.data.
        """
        
        data = np.loadtxt(data_file, dtype = float)
        RegressionDataset.__init__(self, data[:,:-1], data[:,-1], **kwargs)
