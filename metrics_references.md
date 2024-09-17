# References and details on metrics
Reference/documentation for all new metrics that are added but not discussed in the arXiv paper. In the future we will add the metrics from the paper here as well.
## Included metrics overview
In the following we provide a list of metrics that are included in the SynthEval library. We include the relevant reference to where we found the metric, and a brief description of the metric.


### Quantile MSE
Quantile MSE measures the mean squared error of the **10%** percent quantiles of the synthetic data as dictated by the real data. This metric is used to evaluate the distribution of the synthetic data. The metric is calculated as follows:

$
\text{qMSE} = \frac{1}{N_{quant}} \sum_{j=1}^{N_{quant}} \left( x_j - \frac{1}{N_{quant}} \right)^2,
$

where $x_j$ is the estimated probability in each of the $N_{quant}$ quantiles. The metric is calculated for each column in the data, and the average is taken over all columns. The metric is used to evaluate the distribution of the synthetic data, and a low value indicates that the synthetic data is a good representation of the real data. 

Reference:
> Butter, A., Diefenbacher, S., Kasieczka, G., Nachman, B., & Plehn, T. (2021). GANplifying event samples. SciPost Physics, 10(6), 139. [10.21468/SciPostPhys.10.6.139](https://doi.org/10.21468/SciPostPhys.10.6.139)