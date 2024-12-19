# Description: Hitting rate metric
# Author: Anton D. Lautrup
# Date: 23-08-2023

from syntheval.metrics.core.metric import MetricClass

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

    def evaluate(self, thres_percent=1/30) -> float | dict:
        """For hitting rate we regard records as similar if the 
        numerical attributes are close enough and the categorical
        attributes are the same.
        
        Args:
            thres_percent (float): The threshold for numericals (multiplyer for range(att))
        
        Returns:
            dict: The results of the metric
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> HR = HittingRate(real, fake, cat_cols=['a'], num_cols=['b'], do_preprocessing=False)
            >>> HR.evaluate()
            {'hit rate': 1.0}
        """

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
    
    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
     
        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            return [{'metric': 'hit_rate', 'dim': 'p', 
                     'val': self.results['hit rate'], 
                     'n_val': 1-self.results['hit rate'], 
                     }]
        else: pass
