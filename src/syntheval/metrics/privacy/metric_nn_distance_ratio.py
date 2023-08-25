# Description: Nearest neighbour distance ratio metric
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np

from ..core.metric import MetricClass

from ...utils.nn_distance import _knn_distance

class NearestNeighbourDistanceRatio(MetricClass):
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
        return 'nndr'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'privacy'

    def evaluate(self, real=None, fake=None) -> dict:
        """ Compute the Nearest Neighbour Distance Ratio (NNDR) between two datasets.
        """
        if (real is None and fake is None):
            real = self.real_data
            fake = self.synt_data
        
        dist = _knn_distance(real, fake, self.cat_cols,2,self.nn_dist)
        dr = list(map(lambda x: x[0] / x[1], np.transpose(dist)))

        self.results = {'avg': np.mean(dr), 'err': np.std(dr,ddof=1)/np.sqrt(len(dr))}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Nearest neighbour distance ratio         :   %.4f  %.4f   |""" % (self.results['avg'], self.results['err'])
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """
        return {'val': [self.results['avg']], 'err': [self.results['err']]}

    def privacy_loss(self) -> tuple:
        """ Extra function for handling privacy loss. I.e. the difference in
        metric from training data to synthetic data compared to test data.
        This measure is only relevant for a select few metrics.
        
        Privacy loss is always treated as a privacy metric.
        
        Returns normalised output and formatted string.
        """
        train_res = self.results
        test_res = self.evaluate(real=self.hout_data,fake=self.synt_data)

        diff     = abs(test_res['avg'] - train_res['avg'])
        err_diff = np.sqrt(test_res['err']**2+train_res['err']**2)

        string = """\
| Privacy loss (diff. in NNDR)             :   %.4f  %.4f   |""" % (diff, err_diff)
        return {'val': [1-diff], 'err': [err_diff]}, string





