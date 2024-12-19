# Description: Mutual information metric and plot
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np
import pandas as pd

from syntheval.metrics.core.metric import MetricClass

from syntheval.utils.plot_metrics import plot_matrix_heatmap
from sklearn.metrics import normalized_mutual_info_score

def _pairwise_attributes_mutual_information(data):
    """Compute normalized mutual information for all pairwise attributes.

    Elements borrowed from: 
    Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
    Presented at: Proceedingsof the 29th International Conference on Scientific and Statistical Database Management; 2017; Chicago.
    [doi:10.1145/3085504.3091117]
    
    Args:
        data (DataFrame): Data
    
    Returns:
        DataFrame : Matrix
    
    Example:
        >>> _pairwise_attributes_mutual_information(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})) # doctest: +NORMALIZE_WHITESPACE
            a    b
        a  1.0  1.0
        b  1.0  1.0
    """
    labs = sorted(data.columns)
    res = (normalized_mutual_info_score(data[cat1].astype(str),data[cat2].astype(str),average_method='arithmetic') for cat1 in labs for cat2 in labs)
    return pd.DataFrame(np.fromiter(res, dtype=float).reshape(len(labs),len(labs)), columns = labs, index = labs)

class MutualInformation(MetricClass):

    def name() -> str:
        """name/keyword to reference the metric"""
        return 'mi_diff'

    def type() -> str:
        """privacy or utility"""
        return 'utility'

    def evaluate(self, axs_lim=(0,1), axs_scale='Blues') -> float | dict:
        """ Function for evaluating the metric
        
        Args:
            axs_lim (tuple): Axis limits (for plotting)
            axs_scale (str): Color scale (for plotting)
        
        Returns:
            dict: Mutual information matrix difference
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> M = MutualInformation(real, fake, do_preprocessing=False, verbose=False)
            >>> M.evaluate()
            {'mutual_inf_diff': 0.0, 'mi_mat_dims': 2}
        """
        r_mi = _pairwise_attributes_mutual_information(self.real_data)
        f_mi = _pairwise_attributes_mutual_information(self.synt_data)

        mi_mat = r_mi - f_mi
        if self.verbose: plot_matrix_heatmap(mi_mat,'Mutual information matrix difference', 'mi', axs_lim, axs_scale)
        
        self.results = {'mutual_inf_diff': np.linalg.norm(mi_mat, ord='fro'),'mi_mat_dims': len(mi_mat)}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval.        
        """
        string = """\
| Pairwise mutual information difference   :   %.4f           |""" % (self.results['mutual_inf_diff'])
        return string

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            n_elements = int(self.results['mi_mat_dims']*(self.results['mi_mat_dims']-1)/2)
            return [{'metric': 'mutual_inf_diff', 'dim': 'u', 
                     'val': self.results['mutual_inf_diff'], 
                     'n_val': 1-self.results['mutual_inf_diff']/n_elements, 
                     }]
        else: pass