# Description: Template script for making new metric classes
# Author: Anton D. Lautrup
# Date: 18-08-2023

from pandas import DataFrame
from abc import ABC, abstractmethod

from ...utils.variable_detection import get_cat_variables
from ...utils.preprocessing import consistent_label_encoding

class MetricClass(ABC):
    """
    The Metric Class defines an abstract method that contains a skeleton of
    some evaluation metric algorithm
    """ 

    def __init__(
            self,
            real_data: DataFrame,
            synt_data: DataFrame,
            hout_data: DataFrame = None,
            cat_cols: list = None,
            num_cols: list = None,
            nn_dist: str = None,
            analysis_target : str = None,
            do_preprocessing: bool = True,
            verbose: bool = True
    ) -> None:
        
        if do_preprocessing:
            if cat_cols is None:
                cat_cols = get_cat_variables(real_data, threshold=10)
                num_cols = [column for column in real_data.columns if column not in cat_cols]
                print('SynthEval: inferred categorical columns...')
                
            CLE = consistent_label_encoding(real_data, synt_data, cat_cols, hout_data)
            real_data = CLE.encode(real_data)
            synt_data = CLE.encode(synt_data)
            if hout_data is not None: hout_data = CLE.encode(hout_data)

        self.real_data = real_data
        self.synt_data = synt_data
        self.hout_data = hout_data
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        self.nn_dist = nn_dist
        self.analysis_target = analysis_target

        self.results = {}

        self.verbose = verbose

        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        """name/keyword to reference the metric"""
        pass

    @staticmethod
    @abstractmethod
    def type() -> str:
        """privacy or utility"""
        pass

    @abstractmethod
    def evaluate(self) -> float | dict:
        """ Function for evaluating the metric"""
        pass

    @abstractmethod
    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |       
        """
        pass

    @abstractmethod
    def normalize_output(self) -> dict:
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

        Leave empty if the metric should not be used in the index calculations."""
        pass
    
    ### Hooks
    def extra_formatted_output(self) -> tuple:
        """ Some metrics may output both privacy and utility results. For keeping 
        these results seperate in the console print, string output can be placed here 
        to be put in the end of the opposite console text output than the metric type 
        specified above.

        """
        pass