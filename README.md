[![PyPI version](https://badge.fury.io/py/syntheval.svg)](https://badge.fury.io/py/syntheval)
[![doctests](https://github.com/schneiderkamplab/syntheval/actions/workflows/doctests.yml/badge.svg)](https://github.com/schneiderkamplab/syntheval/actions/workflows/doctests.yml)

# SynthEval
The SynthEval library is a framework for evaluating the fidelity of tabularised synthetic data compared with real data. Synthetic data is microdata that is artificially generated and thus does not directly correspond to real-world individuals, making it a possible alternative to regular data anonymity. This tool builds on many previous works and compiles them into a single tool to make the evaluation of synthetic data utility and privacy easier for data scientists and researchers alike.

<p align="center">
  <img src="guides/sketch.png" />
</p>

## Latest version
The current version of the tool offers a wide selection of metrics, to evaluate how well your synthetic data aligns on privacy and utility. In the current version several metrics are available, and can be used in preconfigured or custom evaluation reports. The benchmark module enables multiaxis comparison of several synthetic versions of the same dataset in parallel.  

If you use our library in your work, you can reference us by citing our paper:
```
@article{Lautrup2024,
  author = {Lautrup,  Anton D. and Hyrup,  Tobias and Zimek,  Arthur and Schneider-Kamp,  Peter},
  title = {Syntheval: a framework for detailed utility and privacy evaluation of tabular synthetic data},
  publisher = {Springer Science and Business Media LLC},
  journal = {Data Mining and Knowledge Discovery},
  doi = {10.1007/s10618-024-01081-4},
  year = {2024},
  volume = {39},
  number = {1},
}
```

## Installation
Installation with PyPI using
```
pip install syntheval
```

## User guide
In this section, we briefly outline how to run the main analysis. The library is made to be run with two datasets that look similar, i.e. same number of columns, same variable types and same column and variable names, and will raise errors if that is not the case. The data should be supplied as a pandas dataframes. 
In Python the library is accessed and run in the following way;
```python
from syntheval import SynthEval

evaluator = SynthEval(df_real, holdout_dataframe = df_test, cat_cols = class_cat_col)
evaluator.evaluate(df_fake, class_lab_col, presets_file = "full_eval", **kwargs)
```
Where the user supplies <code>df_real, df_test, df_fake</code> as pandas dataframes, the <code>class_cat_col</code> is a complete list of column names (which can be omitted for categoricals to be automatically inferred). Some metrics require a target class, so <code>class_lab_col</code> is a string (or list or [AnalysisConfig](guides/syntheval_guide.ipynb#Analysis-Target-Configuration) object) for designating variables as a target for downstream task prediction and plotting colouration. In the evaluate function, a presets file can be chosen ("full_eval", "fast_eval", or "privacy") or alternatively, a filepath can be supplied to a json file with select metrics keywords. Finally, instead of (or in addition to), keyword arguments can be added in the end with additional metrics and their options. 

Version 1.4 introduced the benchmark module, that allows a directory of synthetic datasets to be specified for evaluation (or a dictionary of dataframes). All datasets in the folder are evaluated against the training (and test) data on the selected metrics. Three types of rank-derived scoring are available to choose between ("linear", "normal", or "quantile"), assisting in identifying datasets that perform well overall, and on utility and privacy dimensions.
```python
evaluator.benchmark('local/path_to/target_dir/', class_lab_col, presets_file = "full_eval", rank_strategy='normal', **kwargs)
```
Linear ranking is appropriate for datasets where the results are spaced out, e.g. when the synthetic datasets are generated with different methods. Normal ranking works for identifying the datasets that does best/worst out of a normally distributed mass (as you would expect during hyperparameter tuning). Quantile ranking is suitable for benchmarks of several datasets, where multiple winners may be selected from each metric.

For more details on how to use the library, see the codebooks below;
| Notebook | Description |
| --- | --- |
| [Tutorial 1](guides/syntheval_guide.ipynb) | Get started, basic examples |
| [Tutorial 2](guides/syntheval_benchmark.ipynb) | Dataset benchmark, evaluating and ranking synthetic datasets in bulk |
| [Tutorial 3](https://github.com/schneiderkamplab/syntheval-model-benchmark-example/blob/main/syntheval_model_benchmark.ipynb) | Model benchmark example, evaluating and ranking models |
| --- | --- |
| [Preprocessing Reference](guides/preprocessing.md) | Documentation on data preprocessing steps |
| [Metrics Reference](guides/metrics_references.md) | Documentation of the newer metrics that are not covered in the SynthEval paper |

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
The SynthEval library comes equipped with a broad selection of metrics to evaluate various aspects of synthetic tabular data. One of the more interesting properties that makes SynthEval stand out is that many of the metrics have been carefully adapted to accept heterogeneous data. Distances between datapoints are (by default) handled using Gower's distance/similarity measure rather than the Euclidean distance, which negates any requirement of special data encoding.

The metrics are divided into three categories, utility, privacy and fairness, and the results are reported in a standardised format with an average value and an error estimate (where applicable). In addition to using the default preset files, users can select specific metrics and options by using the metric keywords in the evaluate function. The following table gives an overview of the available metrics, their keywords, and links to documentation for each metric.

### Utility Metrics
Utility analysis entails resemblance, quality and usability metrics testing how well the synthetic data looks like, behaves like, and substitutes like the real data.

| keyword | metric name | link to docs | description | 
| --- | --- | --- | --- |
| `dwm` | Dimension-Wise Means | [DimensionWiseMeans](src\syntheval\metrics\utility\metric_dimensionwise_means.py) | nums. only, avg. value and plot |
| `pca` | Principal Components Analysis | [PrincipalComponentsAnalysis](src\syntheval\metrics\utility\metric_principal_component_analysis.py) | [Text Documentation](guides/metrics_references.md#Inter-Dataset-Similarity-Metric-Based-on-PCA) |
| `cio` | Confidence Interval Overlap | [ConfidenceIntervalOverlap](src\syntheval\metrics\utility\metric_confidence_interval_overlap.py) | nums. only, number and fraction of significant tests |
| `corr_diff` | Correlation Matrix Difference | [MixedCorrelation](src\syntheval\metrics\utility\metric_mixed_correlation.py) | mixed correlation |
| `mi_diff` | Mutual Information Matrix Difference | [MutualInformation](src\syntheval\metrics\utility\metric_mutual_information.py) | mixed |
| `ks_test` | Kolmogorov–Smirnov / Total Variation Distance test | [KolmogorovSmirnov](src\syntheval\metrics\utility\metric_kolmogorov_smirnov.py) | avg. distance, avg. p-value and number and fraction of significant tests |
| `h_dist` | Hellinger Distance | [HellingerDistance](src\syntheval\metrics\utility\metric_hellinger_distance.py) | avg. distance |
| `p_MSE` | Propensity Mean Squared Error | [PropensityMeanSquaredError](src\syntheval\metrics\utility\metric_propensity_mse.py) | pMSE and accuracy |
| `auroc_diff` | Prediction AUROC difference | [PredictionAUROCDifference](src\syntheval\metrics\utility\metric_auroc_difference.py) | for binary target variables only |
| `cls_acc`| Classification Accuracy | [ClassificationAccuracy](src\syntheval\metrics\utility\metric_accuracy_difference.py) | avg. TRTR, TSTR across four classifiers, with optional holdout data and 5-fold cross-validation |
| `fio` | Feature Importance Overlap | [FeatureImportanceOverlap](src\syntheval\metrics\utility\metric_feature_importance_overlap.py) | [Text Documentation](guides/metrics_references.md#Feature-Importance-Overlap-(FIO)) |
| `nnaa` | Nearest Neighbour Adversarial Accuracy | [NearestNeighbourAdversarialAccuracy](src\syntheval\metrics\privacy\metric_nn_adversarial_accuracy.py) | avg. NNAA across all records |
| `q_mse` | Quantile MSE | [QuantileMSE](src\syntheval\metrics\utility\metric_quantile_mse.py) | [Text Documentation](guides/metrics_references.md#Quantile-MSE) |
| `mmd` | Maximum Mean Discrepancy (MMD) | [MaximumMeanDiscrepancy](src\syntheval\metrics\utility\metric_max_mean_discrepancy.py) | [Text Documentation](guides/metrics_references.md#Maximum-Mean-Discrepancy-(MMD)) |


### Privacy Metrics
Privacy is a crucial aspect of evaluating synthetic data, we include only three highlevel metrics with more to be added in the future.

| keyword | metric name | link to docs | description | 
| --- | --- | --- | --- |
| `nndr` | Nearest Neighbour Distance Ratio | [NearestNeighbourDistanceRatio](src\syntheval\metrics\privacy\metric_nn_distance_ratio.py) | avg. NNDR across all records |
| `dcr` | Median Distance to Closest Record | [MedianDistanceToClosestRecord](src\syntheval\metrics\privacy\metric_distance_closest_record.py) | normalised by internal NN distance |
| `hit_rate` | Hitting Rate | [HittingRate](src\syntheval\metrics\privacy\metric_hitting_rate.py) | hits on numericals are within attribute range / 30 |
| `eps_risk` | Epsilon Identifiability Risk | [EpsilonIdentifiabilityRisk](src\syntheval\metrics\privacy\metric_epsilon_identifiability.py) | calculated using weighted NN distance |
| `mia` | Membership Inference Attack | [MIAClassifier](src\syntheval\metrics\privacy\metric_MIA_classification.py) | worst case adversarial knowledge attack |
| `att_discl` | Attribute Disclosure Risk | [AttributeDisclosure](src\syntheval\metrics\privacy\metric_AttrDis.py) | with or without holdout data | 

### Fairness Metrics
Fairness is an emerging property of synthetic data, we recently added support to evaluate this aspect, and include for now:

| keyword | metric name | link to docs | description | 
| --- | --- | --- | --- |
| `statistical_parity` | Statistical Parity Difference | [StatisticalParity](src\syntheval\metrics\fairness\metric_statistical_parity.py) | also known as Demographic Parity |

## Creating new metrics
SynthEval is designed with modularity in mind. Creating new, custom metrics is as easy as copying the [metrics template file](https://github.com/schneiderkamplab/syntheval/blob/main/src/syntheval/metrics/metric_template.py), and filling in the five required functions. Because SynthEval has very little hardcoding wrt. the metrics, making new metrics work locally should require no changes other than adding the metrics script in the metrics folder.
