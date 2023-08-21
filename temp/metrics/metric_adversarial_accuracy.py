# Description: Nearest neighbour adversarial accuracy implementation
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np

from .core.metric import MetricClass

from utils.nn_distance import _knn_distance
from sklearn.preprocessing import MinMaxScaler

# def _adversarial_score(real,fake,cat_cols,nn_obj):
#     """Function for calculating adversarial score"""
#     left = np.mean(nn_obj.nn_real_fake()[0] > nn_obj.nn_real_real()[0])
#     right = np.mean(nn_obj.nn_fake_real()[0] > nn_obj.nn_fake_fake()[0])
#     return 0.5 * (left + right)

def _adversarial_score(real, fake, cat_cols, metric):
    """Function for calculating adversarial score"""
    left = np.mean(_knn_distance(real, fake, cat_cols, 1, metric)[0] > _knn_distance(real, real, cat_cols, 1, metric)[0])
    right = np.mean(_knn_distance(fake, real, cat_cols, 1, metric)[0] > _knn_distance(fake, fake, cat_cols, 1, metric)[0])
    return 0.5 * (left + right)

class NearestNeighbourAdversarialAccuracy(MetricClass):
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
        return 'nnaa'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, n_batches=30) -> float | dict:
        """Implementation heavily inspired by original paper"""
        bool_cat_cols = [col1 in self.cat_cols for col1 in self.real_data.columns]

        self.real_data[self.num_cols] = MinMaxScaler().fit_transform(self.real_data[self.num_cols])
        self.synt_data[self.num_cols] = MinMaxScaler().fit_transform(self.synt_data[self.num_cols])
        
        if len(self.real_data)*2 < len(self.synt_data):
            aa_lst = []
            for batch in range(n_batches):
                temp_f = self.synt_data.sample(n=len(self.real_data))
                aa_lst.append(_adversarial_score(self.real_data, temp_f, bool_cat_cols, self.nn_dist))
            self.results = {'avg': np.mean(aa_lst), 'err': np.std(aa_lst, ddof=1)/np.sqrt(len(aa_lst))}
        else:
            self.results = {'avg': _adversarial_score(self.real_data, self.synt_data, bool_cat_cols, self.nn_dist), 'err': 0.0}

        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Nearest neighbour adversarial accuracy   :   %.4f  %.4f   |""" % (self.results['avg'], self.results['err'])
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """
        return {'val': [1-self.results['avg']], 'err': [self.results['err']]}
