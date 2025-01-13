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
> Rajabinasab, M., Lautrup, A.D. & Zimek, A. (2025). Metrics for Inter-Dataset Similarity with Example Applications in Synthetic Data and Feature Selection Evaluation. In Proceedings of the 2025 SIAM International Conference on Data Mining (SDM) (pp. TBD). Society for Industrial and Applied Mathematics. Accepted [20-12-2024] 

### Quantile MSE
Quantile MSE measures the mean squared error of the **10%** percent quantiles of the synthetic data as dictated by the real data. This metric is used to evaluate the distribution of the synthetic data. The metric is calculated as follows:

$$
\text{qMSE} = \frac{1}{N_{quant}} \sum_{j=1}^{N_{quant}} \left( x_j - \frac{1}{N_{quant}} \right)^2,
$$

where $x_j$ is the estimated probability in each of the $N_{quant}$ quantiles. The metric is calculated for each column in the data, and the average is taken over all columns. The metric is used to evaluate the distribution of the synthetic data, and a low value indicates that the synthetic data is a good representation of the real data. 

Reference:
> Butter, A., Diefenbacher, S., Kasieczka, G., Nachman, B., & Plehn, T. (2021). GANplifying event samples. SciPost Physics, 10(6), 139. [10.21468/SciPostPhys.10.6.139](https://doi.org/10.21468/SciPostPhys.10.6.139)