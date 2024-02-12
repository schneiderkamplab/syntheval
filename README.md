[![PyPI version](https://badge.fury.io/py/syntheval.svg)](https://badge.fury.io/py/syntheval)

# SynthEval
The SynthEval library is a framework for evaluating the fidelity of tabularised synthetic data compared with real data. Synthetic data is microdata that is artificially generated and thus does not directly correspond to real-world individuals, making it a possible alternative to regular data anonymity. This tool builds on many previous works, and compile them in a single tool to make evaluation of synthetic data utility and privacy easier for data scientists and reasearchers alike.

## Latest version
The current version of the tool offers a wide selection of utility metrics, to evaluate how well your synthetic data aligns on privacy and utility. In the current several metrics are available, and can be used in preconfigured or custum evaluation reports. The benchmark module enables multiaxis comparison of several synthetic versions of the same dataset in parallel.  

## Installation
Installation with PyPI using
```
pip install syntheval
```

## User guide
In this section we breifly outline how to run the main analysis, for further details see the [notebook](https://github.com/schneiderkamplab/syntheval/blob/main/guides/syntheval_guide.ipynb). The library is made to be run with two datasets that look similar, i.e. same number of columns, same variable types and same column and variable names. The data should be supplied as a pandas dataframes. 
In Python the library is acessed and run in the following way;
```python
from syntheval import SynthEval

evaluator = SynthEval(df_real, holdout_dataframe = df_test, cat_cols = class_cat_col)
evaluator.evaluate(df_fake, class_lab_col, presets_file = "full_eval", **kwargs)
```
Where the user supply <code>df_real, df_test, df_fake</code> as pandas dataframes, the <code>class_cat_col</code> is a complete list of column names (which can be omitted for categoricals to be automatically inferred). Some metrics require a target class, so <code>class_lab_col</code> is a string for designating one column with discrete values as target for usability predictions and coloration. In the evaluate function, a presets file can be chosen ("full_eval", "fast_eval", or "privacy") or alternatively a filepath can be supplied to a json file with select metrics keywords. Finally, instead of (or in addition to), keyword arguments can be added in the end with additional metrics and their options. 

New in version 1.4 is the benchmark module, that allows a directory of synthetic datasets to be specified for evaluation (or a dictionary of dataframes). All datasets in the folder are evaluated against the training (and test) data on the selected metrics. Three types of rank-derived scoring are available to choose between ("linear", "normal", or "quantile"), assisting in identifing datasets that perform well overall, and on utility and privacy dimensions.
```python
evaluator.benchmark('local/path_to/target_dir/', class_lab_col, presets_file = "full_eval", rank_strategy='normal', **kwargs)
```
Linear ranking is appropriate for datasets where the results are spaced out, e.g. when the synthetic datasets are generated with different methods. Normal ranking works for identifying the datasets that does best/worst out of a normally distributed mass (as you would expect during hyperparameter tuning). Quantile ranking is suitable for benchmarks of several datasets, where multiple winners may be selected from each metric.

### Command line interface
SynthEval can also be run from the commandline with the following syntax:
```
> syntheval [OPTIONS] [EVALUATE]

Options:
  -r, --real-data-file PATH     Path to csv file with real data.
  -s, --synt-data-file PATH     Path to csv file with synthetic data.
  -h, --test-data-file PATH     Path to csv file with real data that wasn't
                                used for training.
  -j, --evaluation-config TEXT  Name of preset file or filepath to custom json
                                config file.
  -l, --category-labels PATH    Path to txt file with comma separated labels.
  -c, --class-label TEXT        Label to use for prediction usability and
                                coloring on plots.
  --help                        Show this message and exit.
```

## Included metrics overview
The SynthEval library comes equipped with a broard selection of metrics to evaluate various aspects of synthetic tabular data. Some of the more intresting properties that makes SynthEval stand out is that many of the metrics have been carefully adapted to accept heterogeneous data. Distances between datapoints is (by default) handled using Gower's distance/similarity measure rather than the eucledian distance, which negates any requirement of special data encoding.

### Utility Metrics
Utility analysis entails resemblace, quality and usability metrics testing how well the synthetic data looks like, behaves like, and substitutes like the real data.

In the code we implemented:
- Dimension-Wise Means (nums. only, avg. value and plot)
- Principal Components Analysis (nums. only, plot of first two components)
- Confidence Interval Overlap (nums. only, number and fraction of significant tests)
- Correlation Matrix Difference (mixed correlation)
- Mutual Information Matrix Difference
- Kolmogorov–Smirnov / Total Variation Distance test (avg. distance, avg. p-value and number and fraction of significant tests)
- Hellinger Distance (avg. distance)
- Propensity Mean Squared Error (pMSE and accuracy)
- Prediction AUROC difference (for binary target variables only)
- Nearest Neighbour Adversarial Accuracy (NNAA) 

### classification accuracy
In this tool we test useability by training four different <code>sklearn</code> classifiers on real and synthetic data with 5-fold cross-validation (testing both models on the real validation fold). 
- DecisionTreeClassifier
- AdaBoostClassifier
- RandomForestClassifier
- LogisticRegression

The average accuracy is reported together with the accuracy difference from models trained on real and synthetic data. If a test set is provided, the classifiers are also trained once on the entire training set, and again the accuracy and accuracy differences are reported, but now on the test data.

By default the results are given in terms of accuracy (micro F1 scores). To change, use ‘micro’, ‘macro’ or ‘weighted’ in the preset file or in kwargs.

### Privacy Metrics
Privacy is a crucial aspect of evaluating synthetic data, we include only three highlevel metrics with more to be added in the future.
- Nearest Neighbour Distance Ratio (NNDR)
- Privacy Losses (difference in NNAA and NNDR between test and training sets, good for checking overfitting too.)
- Median Distance to Closest Record (normalised by internal NN distance.)
- Hitting Rate (for nummericals defined to be within the attribute range / 30)
- Epsilon identifiability risk (calculated using weighted NN distance)
- Membership Inference Attack
- Attribute Disclosure Risk (with or without holdout data)

## Creating new metrics
SynthEval is designed with modularity in mind. Creating new, custom metrics is as easy as copying the [metrics template file](https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/metric_template.py), and filling in the five required functions. Because, SynthEval has very little hardcoding wrt. the metrics, making new metrics work locally should require no changes other than adding the metrics script in the metrics folder.