# Description: Nearest neighbour distance ratio metric
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np

from syntheval.metrics.core.metric import MetricClass

from syntheval.utils.nn_distance import _knn_distance

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

    def evaluate(self) -> dict:
        """ Compute the Nearest Neighbour Distance Ratio (NNDR) between two datasets.

        Returns:
            dict: Average NNDR and standard error of the mean
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> NNDR = NearestNeighbourDistanceRatio(real, fake, cat_cols=[], nn_dist='euclid', do_preprocessing=False)
            >>> NNDR.evaluate() # doctest: +ELLIPSIS
            {'avg': 0.66..., 'err': ...}
        """
        dist = _knn_distance(self.real_data, self.synt_data, self.cat_cols, 2, self.nn_dist)
        dr = list(map(lambda x: x[0] / (x[1]+1e-16), np.transpose(dist)))

        self.results = {'avg': float(np.mean(dr)), 'err': float(np.std(dr,ddof=1)/np.sqrt(len(dr)))}

        if self.hout_data is not None:
            dist_h = _knn_distance(self.hout_data, self.synt_data, self.cat_cols, 2, self.nn_dist)
            dr_h = list(map(lambda x: x[0] / (x[1]+1e-16), np.transpose(dist_h)))
            diff     = np.mean(dr_h) - self.results['avg']
            err_diff = np.sqrt((np.std(dr_h,ddof=1)/np.sqrt(len(dr_h)))**2+self.results['err']**2)

            self.results['priv_loss'] = float(diff)
            self.results['priv_loss_err'] = float(err_diff)
        return self.results

    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        rows = []
        rows.append(("privacy",
                    "Nearest neighbour distance ratio",
                    self.results['avg'],
                    self.results['err']))
        if (self.results != {} and self.hout_data is not None):
            rows.append(("privacy",
                        "Privacy loss (diff. in NNDR)",
                        self.results['priv_loss'],
                        self.results['priv_loss_err']))
        return rows

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0 
        """
        if self.results != {}:
            output = [{'metric': 'avg_nndr', 'dim': 'p', 
                     'val': self.results['avg'], 
                     'err': self.results['err'], 
                     'n_val': self.results['avg'], 
                     'n_err': self.results['err'], 
                     }]
            if self.hout_data is not None:
                output.extend([{'metric': 'priv_loss_nndr', 'dim': 'p', 
                     'val': self.results['priv_loss'], 
                     'err': self.results['priv_loss_err'], 
                     'n_val': 1-abs(self.results['priv_loss']), 
                     'n_err': self.results['err'], 
                     }])
            return output
        else: pass