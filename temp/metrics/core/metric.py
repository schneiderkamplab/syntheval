
from pandas import DataFrame
from abc import ABC, abstractmethod

class MetricClass(ABC):
    """
    The Metric Class defines a template method that contains a skeleton of
    some evaluation metric algorithm
    """ 

    def __init__(
            self,
            real_data: DataFrame,
            synt_data: DataFrame,
            hout_data: DataFrame = None,
            cat_cols: list = None,
            num_cols: list = None,
    ) -> None:
        
        self.real_data = real_data
        self.synt_data = synt_data
        self.hout_data = hout_data
        self.cat_cols = cat_cols
        self.num_cols = num_cols

        pass
        
    @staticmethod
    @abstractmethod
    def name() -> str:
        """the name of the metric"""
        ...

    