# Description: Distance to closest record metric
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np

from syntheval.metrics.core.metric import MetricClass

from syntheval.utils.nn_distance import _knn_distance

class MedianDistanceToClosestRecord(MetricClass):
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
        return 'dcr'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'privacy'

    def evaluate(self) -> float | dict:
        """Distance to closest record, using the same NN stuff as NNAA
        
        Returns:
            dict: The results of the metric
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> DCR = MedianDistanceToClosestRecord(real, fake, nn_dist='euclid', do_preprocessing=False)
            >>> DCR.evaluate()
            {'mDCR': 1.0}
        """
        
        distances = _knn_distance(self.synt_data,self.real_data,self.cat_cols,1,self.nn_dist)
        in_dists = _knn_distance(self.real_data,self.real_data,self.cat_cols,1,self.nn_dist)

        int_nn = np.median(in_dists)
        mut_nn = np.median(distances)

        if (int_nn == 0 and mut_nn == 0): dcr = 1
        elif (int_nn == 0 and mut_nn != 0): dcr = 0
        else: dcr = mut_nn/int_nn
        self.results = {'mDCR': dcr}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Median distance to closest record        :   %.4f           |""" % (self.results['mDCR'])
        return string

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        
        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0

        Error fields can be empty.
        """
        if self.results != {}:
            return [{'metric': 'median_DCR', 'dim': 'p', 
                     'val': self.results['mDCR'],
                     'n_val': np.tanh(self.results['mDCR']),
                     }]
        else: pass