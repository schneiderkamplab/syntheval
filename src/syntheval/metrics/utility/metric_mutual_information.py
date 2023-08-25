# Description: Mutual information metric and plot
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np
import pandas as pd

from ..core.metric import MetricClass

from ...utils.plot_metrics import plot_matrix_heatmap
from sklearn.metrics import normalized_mutual_info_score

def _pairwise_attributes_mutual_information(data):
    """Compute normalized mutual information for all pairwise attributes.

    Elements borrowed from: 
    Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
    Presented at: Proceedingsof the 29th International Conference on Scientific and Statistical Database Management; 2017; Chicago.
    [doi:10.1145/3085504.3091117]"""

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

    def evaluate(self) -> float | dict:
        """ Function for evaluating the metric"""
        r_mi = _pairwise_attributes_mutual_information(self.real_data)
        f_mi = _pairwise_attributes_mutual_information(self.synt_data)

        mi_mat = r_mi - f_mi
        plot_matrix_heatmap(mi_mat,'Mutual information matrix difference', 'mi')
        
        self.results = {'mutual_inf_diff': np.linalg.norm(mi_mat, ord='fro')}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval.        
        """
        string = """\
| Pairwise mutual information difference   :   %.4f           |""" % (self.results['mutual_inf_diff'])
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err'
        """
        return ({'val': [1-np.tanh(self.results['mutual_inf_diff'])], 'err': [0]})