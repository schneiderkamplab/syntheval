# Description: Kolmogorov–Smirnov test implementation 
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np

from ..core.metric import MetricClass

from collections import Counter
from scipy.stats import permutation_test,ks_2samp

def _total_variation_distance(x,y):
    """Function for calculating the TVD (KS statistic equivalent)"""
    X, Y = Counter(x), Counter(y)
    merged = X + Y

    return np.round(0.5*sum([abs(X[key]/len(x)-Y[key]/len(y)) for key in merged.keys()]),4)

# def _discrete_ks_statistic(x, y):
#     """Function for calculating the KS statistic"""
#     KstestResult = ks_2samp(x,y)
#     return np.round(KstestResult.statistic,4)

def _discrete_ks(x, y, n_perms=1000):
    """Function for doing permutation test of discrete values in the KS test"""
    res = permutation_test((x, y), _total_variation_distance, n_resamples=n_perms, vectorized=False, permutation_type='independent', alternative='greater')

    return res.statistic, res.pvalue

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
    
        Returns:
            Avg. KS dist: dict  - holds avg. and standard error of the mean (SE)
            Avg. KS pval: dict  - holds avg. and SE
            num of sigs : int   - the number of significant tests at sig_lvl
            frac of sigs: float - the fraction of significant tests at sig_lvl   
        """
        n_dists = []
        c_dists = []
        pvals = []
        sig_cols = []
        
        self.sig_lvl = sig_lvl

        for category in self.real_data.columns:
            R = self.real_data[category]
            F = self.synt_data[category]

            if (category in self.cat_cols):
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

        ### Calculate number of significant tests, and fraction of significant tests
        self.results = {'avg stat' : np.mean(n_dists+c_dists), 
                        'stat err' : np.std(n_dists+c_dists,ddof=1)/np.sqrt(len(n_dists+c_dists)),
                        'avg ks'   : np.mean(n_dists),
                        'ks err'   : np.std(n_dists,ddof=1)/np.sqrt(len(n_dists)),
                        'avg tvd'  : np.mean(c_dists),
                        'tvd err'  : np.std(c_dists,ddof=1)/np.sqrt(len(c_dists)),
                        'avg pval' : np.mean(pvals),
                        'pval err' : np.std(pvals,ddof=1)/np.sqrt(len(pvals)),
                        'num sigs' : len(sig_cols),
                        'frac sigs': len(sig_cols)/len(pvals),
                        'sigs cols': sig_cols
                        }

        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        R = self.results
        string = """\
| Kolmogorov–Smirnov / Total Variation Distance test            |
|   -> average combined statistic          :   %.4f  %.4f   |
|       -> avg. Kolmogorov–Smirnov dist.   :   %.4f  %.4f   |
        -> avg. Total Variation Distance   :   %.4f  %.4f   |
|   -> average combined p-value            :   %.4f  %.4f   |
|       -> # significant tests at a=%.2f   :   %2d               |
|       -> fraction of significant tests   :   %.4f           |""" % (R['avg stat'], R['stat err'],
                                                                      R['avg ks'], R['ks err'],
                                                                      R['avg tvd'], R['tvd err'],
                                                                      R['avg pval'], R['pval err'], 
                                                                      self.sig_lvl, R['num sigs'],
                                                                      R['frac sigs'])
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
