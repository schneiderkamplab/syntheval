# Description: Dimensionwise means difference
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np
import pandas as pd

from ..core.metric import MetricClass

from scipy.stats import sem
from sklearn.preprocessing import MinMaxScaler

from ...utils.plot_metrics import plot_dimensionwise_means

class MetricClassName(MetricClass):
    """The Metric Class is an abstract class that interfaces with 
    SynthEval. When initialised the class has the following attributes:

    Attributes:
    self.real_data : DataFrame
    self.synt_data : DataFrame
    self.hout_data : DataFrame
    self.cat_cols  : list of strings
    self.num_cols  : list of strings

    self.nn_dist   : string keyword
    self.analysis_target: variable name
    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'dwm'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self) -> float | dict:
        """Function for calculating DWM, plotting an appropriate diagram"""

        scaler = MinMaxScaler().fit(pd.concat((self.real_data[self.num_cols],self.synt_data[self.num_cols]),axis=0))
        r_scaled = scaler.transform(self.real_data[self.num_cols])
        f_scaled = scaler.transform(self.synt_data[self.num_cols])

        dim_means = np.array([np.mean(r_scaled,axis=0),np.mean(f_scaled,axis=0)]).T
        means_diff = dim_means[:,0]-dim_means[:,1]
        
        mean_errors = np.array([sem(r_scaled),sem(f_scaled)]).T
        diff_error = np.sqrt(np.sum(mean_errors**2,axis=1))

        if self.verbose: plot_dimensionwise_means(dim_means, mean_errors, self.num_cols)
        self.results = {'avg': np.mean(abs(means_diff)), 'err': np.sqrt(sum(diff_error**2))/len(diff_error)}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Average dimensionwise means diff. (nums) :   %.4f  %.4f   |""" % (self.results['avg'], self.results['err'])
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """

        val_non_lin     = np.exp(-20*self.results['avg'])
        val_non_lin_err = 20*val_non_lin*self.results['err']

        return {'val': [val_non_lin], 'err': [val_non_lin_err]}
