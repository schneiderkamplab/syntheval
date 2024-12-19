# Description: Template script for making new metric classes
# Author: Anton D. Lautrup
# Date: 21-08-2023

from typing import List, Dict, Union

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
        """ Set to 'privacy', 'utility' or 'fairness' """
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

    def normalize_output(self) -> List[dict]:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        This format is required for the metric to be used in the benchmark module. 
        
        The required format is:

        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0

        Error fields can be empty.

        dim, is for designating if metric covers utility ('u'), privacy ('p') or fairness ('f').

        [result] First set of val and err is the regular result found in the 
        results dictionary.

        [Benchmark] Second set is normalised to the zero-one interval so zero 
        represents the worst possible performance and one is the best possible 
        performance."""
        
        if self.results != {}:

            return [{'metric': 'placeholder', 'dim': 'u', 'val': 0.0, 'err': 0.0, 'n_val': 0.0, 'n_err': 0.0}]
        else: pass


    ### Hooks (Extra functions, not required)
    def extra_formatted_output(self) -> Dict[str, str]:
        """ Some metrics may output both utility, privacy and fairness results. 
        For keeping these results seperate in the console print, strings can be 
        placed into a dictionary with appropriate keys to be sent to the correct
        output. Keep the same string formatting as the format_output function.
|                                          :                    |""" 
        pass





