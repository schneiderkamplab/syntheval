# Description: Nearest neighbour adversarial accuracy implementation
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np

from .core.metric import MetricClass

from utils.nn_distance import _knn_distance
from sklearn.preprocessing import MinMaxScaler

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

    def evaluate(self, real=None, fake=None, n_batches=30) -> dict:
        """Implementation heavily inspired by original paper"""
        if (real is None and fake is None):
            real = self.real_data
            fake = self.synt_data
        
        bool_cat_cols = [col1 in self.cat_cols for col1 in real.columns]

        real[self.num_cols] = MinMaxScaler().fit_transform(real[self.num_cols])
        fake[self.num_cols] = MinMaxScaler().fit_transform(fake[self.num_cols])
        
        if len(real)*2 < len(fake):
            aa_lst = []
            for batch in range(n_batches):
                temp_f = fake.sample(n=len(real))
                aa_lst.append(_adversarial_score(real, temp_f, bool_cat_cols, self.nn_dist))
            self.results = {'avg': np.mean(aa_lst), 'err': np.std(aa_lst, ddof=1)/np.sqrt(len(aa_lst))}
        else:
            self.results = {'avg': _adversarial_score(real, fake, bool_cat_cols, self.nn_dist), 'err': 0.0}

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

    def privacy_loss(self,n_batches=30) -> tuple:
        train_res = self.results
        test_res = self.evaluate(real=self.hout_data,fake=self.synt_data,n_batches=n_batches)

        diff     = abs(test_res['avg'] - train_res['avg'])
        err_diff = np.sqrt(test_res['err']**2+train_res['err']**2)

        string = """\
| Privacy loss (diff in NNAA)              :   %.4f  %.4f   |""" % (diff, err_diff)
        return {'val': [1-diff], 'err': [err_diff]}, string
