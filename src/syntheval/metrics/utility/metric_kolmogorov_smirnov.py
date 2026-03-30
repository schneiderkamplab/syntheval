# Description: Kolmogorov–Smirnov test implementation 
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np

from syntheval.metrics.core.metric import MetricClass

from collections import Counter
from scipy.stats import permutation_test, ks_2samp

from syntheval.utils.plot_metrics import plot_significantly_dissimilar_variables

def _total_variation_distance(x,y):
    """Function for calculating the TVD (KS statistic equivalent)
    
    Args:
        x (array-like): Real data
        y (array-like): Synthetic data
    
    Returns:
        float : Total variation distance
    
    Example:
        >>> _total_variation_distance([1,2,3,4,5],[1,2,3,4,5])
        0.0
    """
    X, Y = Counter(x), Counter(y)
    merged = X + Y

    return float(np.round(0.5*sum([abs(X[key]/len(x)-Y[key]/len(y)) for key in merged.keys()]),4))

def _discrete_ks(x, y, n_perms=1000):
    """Function for doing permutation test of discrete values in the KS test
    
    Args:
        x (array-like): Real data
        y (array-like): Synthetic data
        n_perms (int): Number of permutations
    
    Returns:
        float : KS statistic
        float : p-value
    
    Example:
        >>> _discrete_ks([1,2,3,4,5],[1,2,3,4,5])
        (0.0, 1.0)
    """
    res = permutation_test((x, y), _total_variation_distance, n_resamples=n_perms, vectorized=False, permutation_type='independent', alternative='greater')

    return float(res.statistic), float(res.pvalue)

class KolmogorovSmirnovTest(MetricClass):
    """The Metric Class is an abstract class that interfaces with 
    SynthEval. When initialised the class has the following attributes:

    Attributes:
    self.real_data : DataFrame
    self.synt_data : DataFrame
    self.hout_data : DataFrame
    self.cat_cols  : list of strings
    self.num_cols  : list of strings

    self.nn_dist   : string keyword
    
    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'ks_test'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, sig_lvl=0.05, n_perms = 1000) -> float | dict:
        """Function for executing the Kolmogorov-Smirnov test.

        Args:
            sig_lvl (float): Significance level
            n_perms (int): Number of permutations
        
        Returns:
            dict: Average KS statistic and standard error of the mean

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> KST = KolmogorovSmirnovTest(real, fake, cat_cols=['a'], num_cols=['b'], do_preprocessing=False)
            >>> KST.evaluate(sig_lvl=0.05, n_perms=10) # doctest: +ELLIPSIS
            {'avg stat': 0.0, ...}
        """
        n_dists, c_dists = [], []
        pvals = []
        sig_cols = []
        
        self.sig_lvl = sig_lvl

        for category in self.real_data.columns:
            R = self.real_data[category]
            F = self.synt_data[category]

            if category in self.cat_cols:
                statistic, pvalue = _discrete_ks(F,R,n_perms)
                c_dists.append(statistic)
                pvals.append(pvalue)
            else:
                KstestResult = ks_2samp(R,F)
                statistic, pvalue = KstestResult.statistic, KstestResult.pvalue
                n_dists.append(statistic)
                pvals.append(pvalue)
            if pvalue < sig_lvl:
                sig_cols.append(category)

        def _mean_and_se(values):
            if len(values) == 0:
                return np.nan, np.nan
            mean_val = float(np.mean(values))
            if len(values) < 2:
                return mean_val, np.nan
            se_val = float(np.std(values, ddof=1) / np.sqrt(len(values)))
            return mean_val, se_val

        avg_ks, err_ks = _mean_and_se(n_dists)
        avg_tvd, err_tvd = _mean_and_se(c_dists)
        avg_stat, err_stat = _mean_and_se(n_dists + c_dists)
        avg_pval, err_pval = _mean_and_se(pvals)

        ### Calculate number of significant tests, and fraction of significant tests
        self.results = {'avg stat' : float(avg_stat), 'stat err' : float(err_stat),
                        'avg ks'   : float(avg_ks), 'ks err'   : float(err_ks),
                        'avg tvd'  : float(avg_tvd), 'tvd err'  : float(err_tvd),
                        'avg pval' : float(avg_pval), 'pval err' : float(err_pval),
                        'num sigs' : len(sig_cols),
                        'frac sigs': float(len(sig_cols)/len(pvals)),
                        'sigs cols': sig_cols
                        }

        if (self.plot_figures and sig_cols != []): plot_significantly_dissimilar_variables(self.real_data, self.synt_data, sig_cols, self.cat_cols)
        return self.results
    
    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        rows = [
            ("utility", "General Kolmogorov–Smirnov Statistic", self.results['avg stat'], self.results['stat err']),
            ("utility", "  -> Avg. Kolmogorov–Smirnov dist.", self.results['avg ks'], self.results['ks err']),
            ("utility", "  -> Avg. Total Variation Distance", self.results['avg tvd'], self.results['tvd err']),
            ("utility", "Fraction of Significant KS Tests", self.results['frac sigs'], None),
            ("utility", f"  -> # of Significant Tests at a={self.sig_lvl:.2f}", self.results['num sigs'], None),
            ("utility", f"  -> Avg. combined p-value", self.results['avg pval'], self.results['pval err']),
        ]
        return rows
    
#     def format_output(self) -> str:
#         """ Return string for formatting the output, when the
#         metric is part of SynthEval. 
# |                                          :                    |"""
#         R = self.results
#         if self.results != {}:
#             string = """\
# | Kolmogorov–Smirnov / Total Variation Distance test            |
# |   -> average combined statistic          :   %.4f  %.4f   |
# |       -> avg. Kolmogorov–Smirnov dist.   :   %.4f  %.4f   |
# |       -> avg. Total Variation Distance   :   %.4f  %.4f   |
# |   -> average combined p-value            :   %.4f  %.4f   |
# |       -> # significant tests at a=%.2f   :   %2d               |
# |       -> fraction of significant tests   :   %.4f           |""" % (R['avg stat'], R['stat err'],
#                                                                       R['avg ks'], R['ks err'],
#                                                                       R['avg tvd'], R['tvd err'],
#                                                                       R['avg pval'], R['pval err'], 
#                                                                       self.sig_lvl, R['num sigs'],
#                                                                       R['frac sigs'])
#             return string
#         else: pass

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            R = self.results

            return [{'metric': 'ks_tvd_stat', 'dim': 'u', 
                     'val': R['avg stat'], 
                     'err': R['stat err'], 
                     'n_val': 1-R['avg stat'], 
                     'n_err': R['stat err'], 
                     },
                     {'metric': 'frac_ks_sigs', 'dim': 'u', 
                     'val': R['frac sigs'], 
                     'n_val': 1-R['frac sigs'], 
                     }]
        else: pass
