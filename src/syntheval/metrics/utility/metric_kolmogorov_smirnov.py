# Description: Kolmogorov–Smirnov test implementation 
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np

from ..core.metric import MetricClass

from scipy.stats import permutation_test,ks_2samp

def _discrete_ks_statistic(x, y):
    """Function for calculating the KS statistic"""
    KstestResult = ks_2samp(x,y)
    return np.round(KstestResult.statistic,4)

def _discrete_ks(x, y, n_perms=1000):
    """Function for doing permutation test of discrete values in the KS test"""
    res = permutation_test((x, y), _discrete_ks_statistic, n_resamples=n_perms, vectorized=False, permutation_type='independent', alternative='greater')

    return res.statistic, res.pvalue

def featurewise_ks_test(real, fake, cat_cols, sig_lvl=0.05, do_permutation = True, n_perms = 1000):
    """Function for executing the Kolmogorov-Smirnov test.
    
    Returns:
        Avg. KS dist: dict  - holds avg. and standard error of the mean (SE)
        Avg. KS pval: dict  - holds avg. and SE
        num of sigs : int   - the number of significant tests at sig_lvl
        frac of sigs: float - the fraction of significant tests at sig_lvl   
     """
    dists = []
    pvals = []

    for category in real.columns:
        R = real[category]
        F = fake[category]

        if (category in cat_cols and do_permutation==True):
            statistic, pvalue = _discrete_ks(F,R,n_perms)
            dists.append(statistic)
            pvals.append(pvalue)
        else:
            KstestResult = ks_2samp(R,F)
            dists.append(KstestResult.statistic)
            pvals.append(KstestResult.pvalue)

    ### Calculate number of significant tests, and fraction of sifnificant tests
    num  = sum([p_val < sig_lvl for p_val in pvals])
    frac = num/len(pvals)

    return {'avg': np.mean(dists), 'err': np.std(dists,ddof=1)/np.sqrt(len(dists))}, {'avg': np.mean(pvals), 'err':np.std(pvals,ddof=1)/np.sqrt(len(pvals))}, num, frac 

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

    def evaluate(self, sig_lvl=0.05, do_permutation = True, n_perms = 1000) -> float | dict:
        """Function for executing the Kolmogorov-Smirnov test.
    
        Returns:
            Avg. KS dist: dict  - holds avg. and standard error of the mean (SE)
            Avg. KS pval: dict  - holds avg. and SE
            num of sigs : int   - the number of significant tests at sig_lvl
            frac of sigs: float - the fraction of significant tests at sig_lvl   
        """
        dists = []
        pvals = []
        sig_cols = []

        for category in self.real_data.columns:
            R = self.real_data[category]
            F = self.synt_data[category]

            if (category in self.cat_cols and do_permutation==True):
                statistic, pvalue = _discrete_ks(F,R,n_perms)
                dists.append(statistic)
                pvals.append(pvalue)
            else:
                KstestResult = ks_2samp(R,F)
                statistic, pvalue = KstestResult.statistic, KstestResult.pvalue
                dists.append(statistic)
                pvals.append(pvalue)
            if pvalue < sig_lvl:
                sig_cols.append(category)

        ### Calculate number of significant tests, and fraction of significant tests

        self.results = {'avg stat' : np.mean(dists), 
                        'stat err' : np.std(dists,ddof=1)/np.sqrt(len(dists)), 
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
| Kolmogorov–Smirnov test                                       |
|   -> avg. Kolmogorov–Smirnov distance    :   %.4f  %.4f   |
|   -> avg. Kolmogorov–Smirnov p-value     :   %.4f  %.4f   |
|   -> # significant tests at a=0.05       :   %2d               |
|   -> fraction of significant tests       :   %.4f           |""" % (R['avg stat'], 
                                                                      R['stat err'], 
                                                                      R['avg pval'], 
                                                                      R['pval err'], 
                                                                      R['num sigs'],
                                                                      R['frac sigs'])
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """
        R = self.results
        return {'val': [1-R['avg stat'], 1- R['frac sigs']], 'err': [R['stat err'], 0]}
