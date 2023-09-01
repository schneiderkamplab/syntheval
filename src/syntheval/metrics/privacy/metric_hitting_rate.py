# Description: Hitting rate metric
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np
from ..core.metric import MetricClass

class HittingRate(MetricClass):
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
        return 'hit_rate'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'privacy'

    def evaluate(self,thres_percent=1/30) -> float | dict:
        """For hitting rate we regard records as similar if the 
        nummerical attributes are within a threshold range(att)/30"""
        self.thres_percent = thres_percent
        thres = thres_percent*(self.real_data.max() - self.real_data.min())
        thres[self.cat_cols] = 0

        hit = 0
        for i, r in self.real_data.iterrows():
            hit += any((abs(r-self.synt_data) <= thres).all(axis='columns'))
        hit_rate = hit/len(self.real_data)
        self.results = {'hit rate': hit_rate}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Hitting rate (%.2f x range(att))         :   %.4f           |""" % (self.thres_percent, self.results['hit rate'])
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """

        val_non_lin = np.exp(-5*self.results['hit rate'])#max(0,-10*self.results['hit rate'])

        return {'val': [val_non_lin], 'err': [0]}
