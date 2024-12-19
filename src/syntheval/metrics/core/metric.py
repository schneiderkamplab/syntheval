# Description: Template script for making new metric classes
# Author: Anton D. Lautrup
# Date: 18-08-2023

from pandas import DataFrame
from typing import List, Dict
from abc import ABC, abstractmethod

from ...utils.variable_detection import get_cat_variables
from ...utils.preprocessing import consistent_label_encoding

class MetricClass(ABC):
    """
    The Metric Class defines an abstract method that contains a skeleton of
    some evaluation metric algorithm

    Args:
        real_data (DataFrame) : Real dataset
        synt_data (DataFrame) : Synthetic dataset
        hout_data (DataFrame) : Holdout dataset (can be empty)
        cat_cols (List[str]) : List of strings
        num_cols (List[str]) : List of strings
        nn_dist (str) : keyword literal for NN module (not used by all metrics)
        analysis_target (str) : target variable name (not used by all metrics)
        do_preprocessing (bool|object) : whether to preprocess the data or module to use for preprocessing
        verbose (bool) : whether to print and plot results
    """ 

    def __init__(
            self,
            real_data: DataFrame,
            synt_data: DataFrame,
            hout_data: DataFrame = None,
            cat_cols: List[str] = None,
            num_cols: List[str] = None,
            nn_dist: str = None,
            analysis_target : str = None,
            do_preprocessing: bool | object = True,
            verbose: bool = True
    ) -> None:
        
        if isinstance(do_preprocessing, (int, bool)) and do_preprocessing == True:
            if cat_cols is None:
                cat_cols = get_cat_variables(real_data, threshold=10)
                num_cols = [column for column in real_data.columns if column not in cat_cols]
                print('SynthEval: inferred categorical columns...')
                
            CLE = consistent_label_encoding(real_data, synt_data, cat_cols, num_cols, hout_data)
            real_data = CLE.encode(real_data)
            synt_data = CLE.encode(synt_data)
            if hout_data is not None: hout_data = CLE.encode(hout_data)
            self.encoder = CLE
        elif not isinstance(do_preprocessing, (int, bool)):
            self.encoder = do_preprocessing

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
        This format is required for the metric to be used in the benchmark module. 
        
        The required format is:

        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0

        Error fields can be empty.

        dim, is for designating if metric covers utility ('u') or privacy ('p').

        [result] First set of val and err is the regular result found in the 
        results dictionary.

        [Benchmark] Second set is normalised to the zero-one interval so zero 
        represents the worst possible performance and one is the best possible 
        performance."""
        pass
    
    ### Hooks
    def extra_formatted_output(self) -> Dict[str, str]:
        """ Some metrics may output both privacy and utility results. For keeping 
        these results seperate in the console print, string output can be placed here 
        to be put in the end of the opposite console text output than the metric type 
        specified above.

        """
        pass