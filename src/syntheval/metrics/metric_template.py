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
        """ Function for evaluating the metric"""

        self.results = {}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
|                                          :   %.4f  %.4f   |""" % (0.0000, 0.0000)
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best. 
        
        Depending on the metric, you may want to check what scores are actually
        realistically possible and adjust this scale using nonlinearities so that 
        values below 0.95 are actually possible and that the metric does not 
        universally drag the average up. See the existing metrics for examples.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """
        return {'val': [0], 'err': [0]}


    ### Hooks (Extra functions, not required)
    def privacy_loss(self) -> tuple:
        """ Extra function for handling privacy loss. I.e. the difference in
        metric from training data to synthetic data compared to test data.
        This measure is only relevant for a select few metrics.
        
        Privacy loss is always treated as a privacy metric.
        
        Returns output, normalised output and formatted string in that order.
        """
        pass





