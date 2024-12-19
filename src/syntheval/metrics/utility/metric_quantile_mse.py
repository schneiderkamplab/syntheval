# Description: Template script for making new metric classes
# Author: Anton D. Lautrup
# Date: 21-08-2023

from syntheval.metrics.core.metric import MetricClass
import numpy as np

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

    self.verbose   : bool (mainly for supressing prints and plots)

    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'q_mse'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, num_quants=10, cat_mse=False) -> float | dict:
        """Function for executing the quantile mse metric.

        Args:
            num_quants (int): Number of quantiles to divide the data into
            cat_mse (bool): Enable categorical mse
    
        Returns:
            dict : holds avg. and standard error of the mean (SE)
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> QMSE = MetricClassName(real, fake, cat_cols=[], num_cols=[], do_preprocessing=False)
            >>> QMSE.evaluate(num_quants=5, cat_mse=True) # doctest: +ELLIPSIS
            {'avg qMSE': ...
        """
        try:
            assert (len(self.num_cols)>=1 or cat_mse)
        except AssertionError:
            print('Error: Quantile mse did not run, no nummerical attributes, or cat_mse not enabled!')
        else:
            qMSE_lst = []
            for category in self.real_data.columns:

                if category in self.cat_cols and cat_mse:
                    # Categorical data
                    real_items = self.real_data[category].unique()

                    synth_frac = np.array([np.sum(self.synt_data[category] == item) for item in real_items]) / len(self.synt_data)
                    real_frac = np.array([np.sum(self.real_data[category] == item) for item in real_items]) / len(self.real_data)

                    qMSE_lst.append(np.mean((synth_frac - real_frac)**2))
                else:
                    # Numerical data
                    quantiles = np.quantile(self.real_data[category], np.linspace(0, 1, num_quants+1))
                    bin_edges = quantiles.tolist()

                    synth_hist, _ = np.histogram(self.synt_data[category], bins=bin_edges)
                    synth_frac = synth_hist / len(self.synt_data)

                    qMSE_lst.append(np.mean((synth_frac - 1/num_quants)**2))
            
            self.results = {'avg qMSE': np.mean(qMSE_lst), 
                            'qMSE err': np.std(qMSE_lst, ddof=1) / np.sqrt(len(qMSE_lst))
                            }
            return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        if self.results != {}:
            string = """\
| Quantile mean squared error (qMSE)       :   %.4f  %.4f   |""" % (
            self.results['avg qMSE'],
            self.results['qMSE err'],
            )
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
            return [{'metric': 'avg_qMSE', 'dim': 'u', 
                     'val': self.results['avg qMSE'], 
                     'err': self.results['qMSE err'], 
                     'n_val': 1-self.results['avg qMSE'], 
                     'n_err': self.results['qMSE err'], 
                     }]
        else: pass
