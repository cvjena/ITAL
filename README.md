# Information-Theoretic Active Learning (ITAL)

![ITAL Toy Example][teaser]

This repository contains the reference implementation of ITAL and the configuration files of the experiments described in the following paper:

> [**Information-Theoretic Active Learning for Content-Based Image Retrieval.**][paper]  
> Björn Barz, Christoph Käding, and Joachim Denzler.  
> German Conference on Pattern Recognition (GCPR), 2018.


## Dependencies

For ITAL itself:

- **Python 3** (tested with 3.5)
- **numpy** (tested with 1.12)
- **numexpr** (tested with 2.6)
- **scipy** (tested with 0.19)
- **tqdm**

For experiments:

- **scikit-learn**
- **scikit-image**
- **matplotlib**


## Using ITAL

### Initialization

Using ITAL for interactive retrieval with relevance feedback is easy.
First, import the main class `ITAL` and instantiate it:

```python
from ital.ital import ITAL
learner = ITAL(data, queries, length_scale = 0.1)
```

Here, `data` refers to the entire dataset as an `n-by-d` matrix containing `n` samples with `d` features and `queries` is a list of query feature vectors.

Instantiating and fitting the learner can also be divided into two steps if you want to re-use the learner with the same configuration:

```python
learner = ITAL(length_scale = 0.1)
learner.fit(data, queries)
```

Remember to always choose an appropriate `length_scale` hyper-parameter for your type of data.

### Retrieving samples according to the current relevance model

To retrieve the top 100 relevant samples from the dataset, given the queries and the feedback given so far, use:

```python
most_relevant = learner.top_results(100)
```

This returns a list of sample indices, sorted by decreasing relevance.

### Fetching samples for annotation and updating the relevance model

For improving the relevance model, we first need to fetch a small batch of samples which we would like the user to annotate regarding their relevance:

```python
candidates = learner.fetch_unlabelled(4)
```

This will obtain the indices of 4 samples which ITAL considers to be most promising.

Then, we can obtain relevance feedback from the user and assign one of the following relevance labels to all the candidates: `1` for relevant samples, `-1` for irrelevant samples, or `0` if the user is uncertain. We store this feedback in a dictionary `feedback` mapping sample indices to the aforementioned relevance labels and update the model as follows:

```python
learner.update(feedback)
```


### Other learners

Besides ITAL, we also provide implementations of several other popular active learning techniques with an identical API as described above in the module `ital.baseline_methods`.
Especially `BorderlineDiversitySampling` and `TCAL` might also be worth a try, since they are the second-best and third-best performers after ITAL, but faster.


## Running Automated Experiments

The script `run_experiment.py` can be used to automatically generate a number of query scenarios and perform active learning with simulated user feedback to benchmark different methods.
It takes a path to a configuration file as argument, which must contain a section `[EXPERIMENT]` specifying the following directives:

- `dataset` (string, required): the type of the dataset to be used corresponding to one of the classes defined in `datasets.py`, but without the `Dataset` suffix (e.g., `USPS` for the `USPSDataset` class).
- `method` (string, required): the name of the active learning method to be used. Possible values can be found among the keys of the dictionary `utils.LEARNERS`.
- `repetitions` (int, default: 10): number of experiments with different random queries per class. Results will be averaged over repetitions.
- `rounds` (int, default: 10): number of iterative feedback rounds.
- `batch_size` (int, required): number of candidates to fetch for each feedback round.
- `num_init` (int, default: 1): number of initial positive samples per query.
- `initial_negatives` (int, default: 0): number of initial negative samples per query.
- `label_prob` (float, default: 1.0): probability for the event that the user gives feedback regarding a particular candidate sample.
- `mistake_prob` (float, default: 0.0): probability that the user provides the wrong label (given that a label is provided at all).
- `query_classes` (string, optional): space-separated list of classes to draw query images from. If not specified, all classes will be used.
- `avg_class_perf` (boolean, default: no): whether to report performance averaged over all classes or for each class individually.

These directives can also be overridden on the command-line by passing `--key=value` arguments to `run_experiment.py`.

In addition, each config file must contain a section with the same name as the value for `dataset` which provides the keyword arguments for the constructor of the dataset interface class.

Similarly, a section with the same name as the value for `method` can be used to specify arguments for the constructor of the active learning method.
Default values to be applied to all learning methods (e.g., hyper-parameters of the GP kernel) can also be specified in a special section `[METHOD_DEFAULTS]`.


## Reproducing the Experiments from the Paper

The results reported in the paper have been obtained by running the script `run_experiment.py` on the configuration files in the `configs` directory, in particular:

- [`configs/butterflies.conf`](configs/butterflies.conf)
- [`configs/usps.conf`](configs/usps.conf)
- [`configs/natural_scenes.conf`](configs/natural_scenes.conf)
- [`configs/mirflickr.conf`](configs/mirflickr.conf)
- [`configs/imagenet.conf`](configs/imagenet.conf)

Everything except ImageNet should work out of the box, since the required features are provided in the `data` directory.
The `MIRFLICKRDataset` can take a path to an image directory, which is not included, but that should only be necessary if you want to plot the individual learning steps.

For the ImageNet experiments, you need to download the [ILSVRC 2010][ilsvrc] devkit and the SBOW features and adjust the paths in the config file accordingly.


[paper]: http://hera.inf-cv.uni-jena.de:6680/pdf/Barz18:ITAL.pdf "PDF"
[teaser]: https://user-images.githubusercontent.com/7915048/44797170-cf935a80-abae-11e8-89c8-fd5329a1de3a.png
[ilsvrc]: http://image-net.org/challenges/LSVRC/2010/index