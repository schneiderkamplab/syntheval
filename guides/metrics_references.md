# References and details on metrics
Reference/documentation for all new metrics that are added but not discussed in the arXiv paper. In the future we will add the metrics from the paper here as well.
## Included metrics overview
In the following we provide a list of metrics that are included in the SynthEval library. We include the relevant reference to where we found the metric, and a brief description of the metric.


### Inter-Dataset Similarity Metric Based on PCA
The metrics 'exp_var_diff' and 'comp_angle_diff' are two global utility metrics distinct from the other metrics in the library. They are derived from the principal component analysis (PCA) of the real and synthetic datasets and based on the idea that if two datasets are similar, their projections onto the principal components should be similar. The metrics are calculated as follows:

$$
\Delta \lambda = \frac{d}{2(d-1)} \sum_i^p |\lambda_i-\lambda_i'| \qquad\qquad \Delta \theta = \frac{2}{\pi}\min \left[\arccos{(\bm a_1\!\cdot\! \bm a_1')}, \arccos{(\bm a_1\!\cdot\!(-\bm a_1'))}\right]
$$

where the first normalisation factor $d/2(d-1)$ is based on the dimensionality of the dataset (used to quantify the maximum discrepancy between the eigenvalues), and the second normalisation factor $2/\pi$ is used to scale the angle difference to the range $[0,1]$. The angle difference metric has the minimum to acount for vectors comming out of the PCA as antiparallel. A lower value of $\Delta \lambda$ and $\Delta \theta$ indicates that the real and synthetic datasets are similar. Note that the metric is rather unstable if using less than a couple hundred samples.

Reference:
> Rajabinasab, M., Lautrup, A.D. & Zimek, A. (2025). Metrics for Inter-Dataset Similarity with Example Applications in Synthetic Data and Feature Selection Evaluation. In Proceedings of the 2025 SIAM International Conference on Data Mining (SDM) (pp. 527--537). Society for Industrial and Applied Mathematics.

### Quantile MSE
Quantile MSE measures the mean squared error of the **10%** percent quantiles of the synthetic data as dictated by the real data. This metric is used to evaluate the distribution of the synthetic data. The metric is calculated as follows:

$$
\text{qMSE} = \frac{1}{N_{quant}} \sum_{j=1}^{N_{quant}} \left( x_j - \frac{1}{N_{quant}} \right)^2,
$$

where $x_j$ is the estimated probability in each of the $N_{quant}$ quantiles. The metric is calculated for each column in the data, and the average is taken over all columns. The metric is used to evaluate the distribution of the synthetic data, and a low value indicates that the synthetic data is a good representation of the real data. 

Reference:
> Butter, A., Diefenbacher, S., Kasieczka, G., Nachman, B., & Plehn, T. (2021). GANplifying event samples. SciPost Physics, 10(6), 139. [10.21468/SciPostPhys.10.6.139](https://doi.org/10.21468/SciPostPhys.10.6.139)

### Maximum Mean Discrepancy (MMD)
MMD is a kernel-based distance measure between two distributions (in our case the real and synthetic data sets). It comes in two flavors: the biased V-statistic and the unbiased U-statistic. The biased V-statistic is calculated as follows:

$$
\text{MMD}_b^2 = \frac{1}{n^2} \sum_{i,j} k(x_i, x_j) + \frac{1}{m^2} \sum_{i,j} k(y_i, y_j) - \frac{2}{nm} \sum_{i,j} k(x_i, y_j),
$$

where $k$ is a kernel function, and $x_i$ and $y_j$ are samples from the two distributions. The unbiased U-statistic is calculated as follows:

$$
\text{MMD}_u^2 = \frac{1}{n(n-1)} \sum_{i \neq j} k(x_i, x_j) + \frac{1}{m(m-1)} \sum_{i \neq j} k(y_i, y_j) - \frac{2}{nm} \sum_{i,j} k(x_i, y_j),
$$

where the sums are taken over all pairs of samples, excluding the diagonal terms. Both measures can be negative at finite sample sizes due to variance, so it is clipped at $0$ for stability, i.e., $\max(MMD^2,0)$. A lower value of MMD indicates that the two distributions are similar (yet negative values cannot be compared by magnitude). 

As a test statistic, MMD can be used to perform a two-sample test to determine if the two distributions are significantly different. In this case, a value higher than some threshold (determined by the distribution of MMD under the null hypothesis) would indicate that the two distributions are significantly different. In the context of synthetic data evaluation, a low MMD value would indicate that the synthetic data is not unreasonably different.

Reference:
> Gretton, A., Borgwardt, K.M., Rasch, M.J., Smola, A., Schölkopf, B., & Smola, A. (2012). A Kernel Two-Sample Test. Journal of Machine Learning Research, 13(25), 723–773. [http://jmlr.org/papers/v13/gretton12a.html](http://jmlr.org/papers/v13/gretton12a.html)

### Feature Importance Overlap (FIO)
FIO is a generic metric that measures the overlap in top-k selected features between a model trained on real data and a model trained on synthetic data in predicting the target analysis variable. The metric is calculated as follows:

$$
\text{FIO}_k = \frac{|\text{Top-$k$ features from real data} \cap \text{Top-$k$ features from synthetic data}|}{k},
$$

In the actual implementation we select top-5%, 10%, 25%, and 50% of the features (where possible), and return all successful results. A higher value of FIO indicates that the synthetic data is a good representation of the real data in terms of feature selection. However, for the lowest top-k selections (e.g., 5% or 10%), it should be *very close* to 1 for a good synthetic data set, while for higher top-k selections (e.g., 25% or 50%) it can be more divergent and still indicate a good synthetic data set.
