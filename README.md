# SynthEval
The SynthEval library is a tool made for evaluating the quality of tabularised synthetic data compared with real data. Synthetic data is microdata that is artificially generated and thus does not directly correspond to real-world individuals, making it a possible alternative to regular data anonymity. This tool builds on many previous works, and compile them in a single tool to make evaluation of synthetic data utility easier for data scientists and reasearchers alike.

## Latest version
The current version of the tool offers a wide selection of utility metrics, to evaluate how well your synthetic data aligns on quality, resemblance and usability. In the current version we include only three high level privacy tools, but it is the aim to provide a more extensive assesment of disclosure risk in a future version. 

## Installation




## User guide
In this section we breifly outline how to run the main test, for further details see the "syntheval_guide.ipynb". The library is made to be run with two datasets that look similar, i.e. same number of columns, same variable types and same column and variable names. The data should be supplied as a pandas dataframe. 
In Python the library is acessed and run in the following way;
```python
from syntheval import SynthEval

evaluator = Syntheval(df_real, hold_out = df_test, cat_cols = class_cat_col)
evaluator.full_eval(df_fake, class_lab_col)
```
Where the user supply <code>df_real, df_test, df_fake</code> as pandas dataframes, as well as the <code>class_cat_col</code> list of column names for the categorical variables and <code>class_lab_col</code> string for designating one column with discrete values as target for usability predictions and coloration. 

Results are saved to a csv file, multiple runs of the same SynthEval instance with different synthetic data files will save new rows allowing for various uses such as snapshots, checkpoints and benchmarking. 

## Included metrics overview
The SynthEval library comes equipped with a broard selection of metrics to evaluate various aspects of synthetic tabular data.

### Quality evaluation
Quality metrics are used for checking if the statistical properties of the real data carries over into the synthetic version. This is mainly done by checking pairwise properties, such as correlation and distributional similarity. 

In the code we implemented:
- Correlation matrix difference (for the nummericals only)
- Pairwise mutual information matrix difference (for all datatypes)
- Kolmogorov–Smirnov test (avg. distance, avg. p-value and number and fraction of significant tests)

### Resemblance evaluation
Resemblance metrics are for assessing if the synthetic data can be distinguished from the real data. While the preliminary tests already are visualizing the data, additional tools are used in checking synthetic data resemblance. We include:
- Confidence interval overlap (average and count of nonoverlaps)
- Hellinger distance (average)
- propensity mean squared error
- Nearest neighbour adversarial accuracy 

### Usability evaluation
Useability is a core attribute of utility, and entails how well the synthetic data can act as a replacement for real data and provide a similar analysis. In this tool we test useability by training four different <code>sklearn</code> classifiers on real and synthetic data with 5-fold cross-validation (testing both models on the real validation fold). 
- DecisionTreeClassifier
- AdaBoostClassifier
- RandomForestClassifier
- LogisticRegression

The average accuracy is reported together with the accuracy difference from models trained on real and synthetic data. If a test set is provided, the classifiers are also trained once on the entire training set, and again the accuracy and accuracy differences are reported, but now on the test data.

By default the results are given in terms of accuracy (micro F1 scores). To change, use {‘micro’, ‘macro’, ‘weighted’} for the <code>SynthEval.F1_type</code> attribute.

### Utility score
finally, a summary utility score is calculated based on the tests described above. Specifically we calculate the utility score in the following way
$$\mathrm{UTIL} = \frac{1}{10} [ (1-\tanh{\mathrm{corr. diff.}})+(1-\tanh{\mathrm{MI diff.}})+ (1-\mathrm{KS dist.}) + (1-\mathrm{KS sig.frac.}) + \mathrm{CIO}+ (1-\mathrm{H dist.}) + \left(1-\frac{\mathrm{pMSE}}{0.25}\right) +(1-\mathrm{NNAA})+ (1-\mathrm{train F1 diff.})+(1-\mathrm{test F1 diff.})]$$

### Privacy evaluation
Privacy is a crucial aspect of evaluating synthetic data, we include only three highlevel metrics with more to be added in the future.
- average distance to closest record (normed, and divided by avg. NN dist)
- hitting rate (for nummericals defined to be within the attribute range / 30)
- privacy loss (difference in NNAA between test and training set, also works for checking overfitting)
