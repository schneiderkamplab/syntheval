# Description: Script for calculating the mixed correlation
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np
import pandas as pd

from syntheval.metrics.core.metric import MetricClass

from scipy.stats import chi2_contingency
from syntheval.utils.plot_metrics import plot_matrix_heatmap

def _cramers_V(var1,var2) :
    """function for calculating Cramers V between two categorial variables
    credit: https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix

    Args:
        var1 (array-like): Real data
        var2 (array-like): Synthetic data
    
    Returns:
        float : Cramers V
    
    Example:
        >>> _cramers_V([1,2,3,4,5],[1,2,3,4,5])
        1.0...
    """
    crosstab =np.array(pd.crosstab(var1, var2, rownames=None, colnames=None)) # Cross table building
    stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
    return (stat/(obs*mini+1e-16))

def _apply_mat(data,func,labs1,labs2):
    """Help function for constructing a matrix based on func accross labels 1 and 2
    
    Args:
        data (DataFrame): Data
        func (function): Function to apply
        labs1 (list): Labels 1
        labs2 (list): Labels 2

    Returns:
        DataFrame : Matrix
    
    Example:
        >>> _apply_mat(pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}), _cramers_V ,['a','b'], ['a','b']) # doctest: +NORMALIZE_WHITESPACE
            a    b
        a  1.0  1.0
        b  1.0  1.0
    """
    res = (func(data[lab1],data[lab2]) for lab1 in labs1 for lab2 in labs2)
    return pd.DataFrame(np.fromiter(res, dtype=float).reshape(len(labs1),len(labs2)), columns = labs2, index = labs1)

def _correlation_ratio(categories, measurements):
    """Function for calculating the correlation ration eta^2 of categorial and nummerical data
    
    Args:
        categories (array): Categories
        measurements (array): Measurements
    
    Returns:
        float : Eta^2
    
    Example:
        >>> _correlation_ratio(np.array([0,1,0,1]),np.array([1,2,3,4]))
        0.2
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[fcat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

def mixed_correlation(data,num_cols,cat_cols):
    """Function for calculating a correlation matrix of mixed datatypes.
    Spearman's rho is used for rank-based correlation, Cramer's V is used for categorical variables, 
    and correlation ratio is used for categorical and continuous variables.

    Args:
        data (DataFrame): Data
        num_cols (list): Numerical columns
        cat_cols (list): Categorical columns

    Returns:
        DataFrame : Correlation matrix

    Example:
        >>> mixed_correlation(pd.DataFrame({'num': [1, 2, 3], 'cat': [4, 5, 6]}),['num'],['cat']) # doctest: +NORMALIZE_WHITESPACE
            cat  num
        cat  1.0  1.0
        num  1.0  1.0
    """
    corr_num_num = data[num_cols].corr()
    corr_cat_cat = _apply_mat(data,_cramers_V,cat_cols,cat_cols)
    corr_cat_num = _apply_mat(data,_correlation_ratio,cat_cols,num_cols)
    if corr_cat_cat.empty: corr = corr_num_num
    elif corr_num_num.empty: corr = corr_cat_cat
    else:
        top_row = pd.concat([corr_cat_cat,corr_cat_num],axis=1)
        bot_row = pd.concat([corr_cat_num.transpose(),corr_num_num],axis=1)
        corr = pd.concat([top_row,bot_row],axis=0)
    return corr + np.diag(1-np.diag(corr))

class MixedCorrelation(MetricClass):
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
        return 'corr_diff'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, mixed_corr=True, return_mats=False, axs_lim=(-1,1), axs_scale="RdBu") -> dict:
        """Function for calculating the (mixed) correlation matrix difference.
        This calculation uses spearmans rho for numerical-numerical, Cramer's V for categories,
        and correlation ratio (eta) for numerical-categorials.
                
        Args:
            mixed_corr (bool): Use mixed correlation
            return_mats (bool): Return the individual correlation matrices
            axs_lim (tuple): Axis limits (for plotting)
            axs_scale (str): Axis scale (for plotting)

        Returns:
            dict: Frobenius norm of the correlation matrix difference
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'num': [1, 2, 3], 'cat': [0, 1, 0]})
            >>> fake = pd.DataFrame({'num': [1, 2, 3], 'cat': [0, 1, 0]})
            >>> MC = MixedCorrelation(real, fake, cat_cols=['cat'], num_cols=['num'], do_preprocessing=False, verbose=False)
            >>> MC.evaluate(mixed_corr=True)
            {'corr_mat_diff': 0.0, 'corr_mat_dims': 2}
        """
        self.mixed_corr = mixed_corr
        if mixed_corr:
            r_corr = mixed_correlation(self.real_data,self.num_cols,self.cat_cols)
            f_corr = mixed_correlation(self.synt_data,self.num_cols,self.cat_cols)
            corr_mat = r_corr-f_corr
            if self.verbose: plot_matrix_heatmap(corr_mat,'Mixed correlation matrix difference', 'corr', axs_lim, axs_scale)
        else:
            r_corr = self.real_data[self.num_cols].corr()
            f_corr = self.synt_data[self.num_cols].corr()
            corr_mat = r_corr-f_corr
            if self.verbose: plot_matrix_heatmap(corr_mat,'Correlation matrix difference (nums only)', 'corr', axs_lim, axs_scale)
        
        self.results = {'corr_mat_diff': np.linalg.norm(corr_mat,ord='fro'), 'corr_mat_dims': len(corr_mat)}
        if return_mats: self.results['real_cor_mat'] = r_corr
        if return_mats: self.results['synt_cor_mat'] = f_corr
        if return_mats: self.results['diff_cor_mat'] = corr_mat
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        if self.mixed_corr:
            string = """\
| Mixed correlation matrix difference      :   %.4f           |""" % (self.results['corr_mat_diff'])
        else:
            string = """\
| Correlation difference (nums only)       :   %.4f           |""" % (self.results['corr_mat_diff'])
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
            n_elements = int(self.results['corr_mat_dims']*(self.results['corr_mat_dims']-1)/2)
            return [{'metric': 'corr_mat_diff', 'dim': 'u', 
                     'val': self.results['corr_mat_diff'], 
                     'n_val': 1-self.results['corr_mat_diff']/n_elements, 
                     }]
        else: pass
