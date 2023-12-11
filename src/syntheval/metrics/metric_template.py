# Description: Template script for making new metric classes
# Author: Anton D. Lautrup
# Date: 21-08-2023

from .core.metric import MetricClass

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
        return 'temp'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self) -> float | dict:
        """ Function for evaluating the metric and returning raw results"""

        self.results = {}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
|                                          :   %.4f  %.4f   |""" % (0.0000, 0.0000)
        return string

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        This format is required for the metric to be used in the benchmark module, 
        and also if the metric is to contribute to the utility or privacy index. 
        
        The required format is:

        metric  dim  val  err  n_val  n_err idx_val idx_err
            name1  u  0.0  0.0    0.0    0.0    None    None
            name2  p  0.0  0.0    0.0    0.0    0.0     0.0

        Error fields and idx can be empty.

        dim, is for designating if metric covers utility ('u') or privacy ('p').

        [result] First set of val and err is the regular result found in the 
        results dictionary.

        [Benchmark] Second set is normalised to the zero-one interval so zero 
        represents the worst possible performance and one is the best possible 
        performance. 

        [Indecies] Used for aggregating results of different metrics to a combined
        utility or privacy index. Depending on the metric, you may want to check 
        what scores are actually realistically possible and adjust this scale using 
        nonlinearities so that values below say 0.95 are actually possible and that 
        the metric does not drown out signal from other metrics. See the existing 
        metrics for examples.

        Leave empty if the metric should not be used in the index calculations.   
        """
        if self.results != {}:

            return [{'metric': 'placeholder', 'dim': 'u', 'val': 0.0, 'err': 0.0, 'n_val': 0.0, 'n_err': 0.0, 'idx_val': 0.0, 'idx_err': 0.0}]
        else: pass


    ### Hooks (Extra functions, not required)
    def privacy_loss(self) -> tuple:
        """ Extra function for handling privacy loss. I.e. the difference in
        metric from training data to synthetic data compared to test data.
        This measure is only relevant for a select few metrics.
        
        Privacy loss is always treated as a privacy metric.
        
        Returns output, normalised output and formatted string in that order.
        """
        pass





