# Description: Confidence interval overlap
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np

from scipy.stats import sem

from syntheval.metrics.core.metric import MetricClass

class ConfidenceIntervalOverlap(MetricClass):
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
        return 'cio'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self,confidence=95) -> float | dict:
        """Function for calculating the average CIO, also returns the 
        number of nonoverlapping interval
        
        Args:
            confidence (int): Confidence level for the confidence interval
        
        Returns:
            dict: Average confidence interval overlap, overlap error, number of non-overlapping intervals, and fraction of non-overlapping intervals
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> CIO = ConfidenceIntervalOverlap(real, fake, cat_cols=[], num_cols=['a','b'], do_preprocessing=False)
            >>> CIO.evaluate() # doctest: +ELLIPSIS
            {'avg overlap': 1.0, ...}
        """
        confidence_table = {80: 1.28, 90: 1.645, 95: 1.96, 98: 2.33, 99: 2.58}
        try:
            assert len(self.num_cols) > 0
            assert confidence in confidence_table.keys()
        except AssertionError:
            if len(self.num_cols) == 0:
                print(" Warning: No nummerical attributes provided for confidence interval overlap metric.")
            else:
                print(" Error: Confidence level not recognized, choose 80, 90, 95, 98 or 99.")
            return {}
        else:
            self.confidence = confidence

            if confidence in confidence_table.keys():
                z_value = confidence_table[confidence]
                mus = np.array([np.mean(self.real_data[self.num_cols],axis=0),np.mean(self.synt_data[self.num_cols],axis=0)]).T
                sems = np.array([sem(self.real_data[self.num_cols]),sem(self.synt_data[self.num_cols])]).T
                
                CI = sems*z_value
                us = mus+CI
                ls = mus-CI

                Jk = []
                for i in range(len(CI)):
                    top = (min(us[i][0],us[i][1])-max(ls[i][0],ls[i][1]))
                    Jk.append(max(0,0.5*(top/(us[i][0]-ls[i][0])+top/(us[i][1]-ls[i][1]))))

                num = sum([j == 0 for j in Jk])
                frac = num/len(Jk)
                self.results = {'avg overlap': np.mean(Jk), 
                                'overlap err': np.std(Jk,ddof=1)/np.sqrt(len(Jk)), 
                                'num non-overlaps': num, 
                                'frac non-overlaps': frac
                                }
                return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval.
|                                          :                    |"""
        if self.results != {}:
            string = """\
| Average confidence interval overlap      :   %.4f  %.4f   |
|   -> # non-overlapping COIs at %2d%%       :   %2d               |
|   -> fraction of non-overlapping CIs     :   %.4f           |""" % (
            self.results['avg overlap'],
            self.results['overlap err'],
            self.confidence,
            self.results['num non-overlaps'],
            self.results['frac non-overlaps'])
            return string
        else: pass

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            return [{'metric': 'avg_cio', 'dim': 'u', 
                     'val': self.results['avg overlap'], 
                     'err': self.results['overlap err'], 
                     'n_val': self.results['avg overlap'], 
                     'n_err': self.results['overlap err'], 
                     }]
        else: pass