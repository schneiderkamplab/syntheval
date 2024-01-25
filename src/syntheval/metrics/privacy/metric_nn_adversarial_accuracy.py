# Description: Nearest neighbour adversarial accuracy implementation
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np

from ..core.metric import MetricClass

from ...utils.nn_distance import _knn_distance
from sklearn.preprocessing import MinMaxScaler

def _adversarial_score(real, fake, cat_cols, metric):
    """Function for calculating adversarial score"""
    left = np.mean(_knn_distance(real, fake, cat_cols, 1, metric)[0] > _knn_distance(real, real, cat_cols, 1, metric)[0])
    right = np.mean(_knn_distance(fake, real, cat_cols, 1, metric)[0] > _knn_distance(fake, fake, cat_cols, 1, metric)[0])
    return 0.5 * (left + right)

def evaluate_dataset_nnaa(real, fake, num_cols, cat_cols, metric, n_resample):
    """Helper function for running adversarial score multiple times if the 
    datasets have much different sizes."""
    real[num_cols] = MinMaxScaler().fit_transform(real[num_cols])
    fake[num_cols] = MinMaxScaler().fit_transform(fake[num_cols])
    
    real_fake = len(real)/len(fake)
    fake_real = len(fake)/len(real)

    if any([real_fake >= 2, fake_real >= 2]):
        aa_lst = []
        for batch in range(n_resample):
            temp_r = real if real_fake < 2 else real.sample(n=len(fake))
            temp_f = fake if fake_real < 2 else fake.sample(n=len(real))
            aa_lst.append(_adversarial_score(temp_r, temp_f, cat_cols, metric))

        avg = np.mean(aa_lst)
        err = np.std(aa_lst, ddof=1)/np.sqrt(len(aa_lst))
    else:
        avg = _adversarial_score(real, fake, cat_cols, metric)
        err = 0.0

    return avg, err

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

    def evaluate(self, n_resample=30) -> dict:
        """Implementation heavily inspired by original paper"""


        avg, err = evaluate_dataset_nnaa(self.real_data,self.synt_data,self.num_cols,self.cat_cols,self.nn_dist,n_resample)

        self.results = {'avg': avg, 'err': err}

        if self.hout_data is not None:
            avg_h, err_h = evaluate_dataset_nnaa(self.hout_data,self.synt_data,self.num_cols,self.cat_cols,self.nn_dist,n_resample)
            diff = avg_h - avg
            err_diff = np.sqrt(err_h**2+err**2)

            self.results['priv_loss'] = diff
            self.results['priv_loss_err'] = err_diff

        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Nearest neighbour adversarial accuracy   :   %.4f  %.4f   |""" % (self.results['avg'], self.results['err'])
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
            output =  [{'metric': 'nnaa', 'dim': 'u', 
                        'val': self.results['avg'], 
                        'err': self.results['err'], 
                        'n_val': 1-self.results['avg'], 
                        'n_err': self.results['err'], 
                        # 'idx_val': 1-self.results['avg'], 
                        # 'idx_err': self.results['err']
                        }]
            if self.hout_data is not None:
                # val_non_lin     = np.exp(-15*max(0,self.results['priv_loss']))
                # val_non_lin_err = 15*val_non_lin*self.results['priv_loss_err']
                output.extend([{'metric': 'priv_loss_nnaa', 'dim': 'p', 
                        'val': self.results['priv_loss'], 
                        'err': self.results['priv_loss_err'], 
                        'n_val': 1-abs(self.results['priv_loss']), 
                        'n_err': self.results['priv_loss_err'], 
                        # 'idx_val': val_non_lin, 
                        # 'idx_err': val_non_lin_err
                        }])
            return output
        else: pass

    def extra_formatted_output(self) -> str:
        """Bit for printing the privacy loss together with the other privacy metrics"""
        if (self.results != {} and self.hout_data is not None):
            string = """\
| Privacy loss (diff. in NNAA)             :   %.4f  %.4f   |""" % (self.results['priv_loss'], self.results['priv_loss_err'])
            return string
        else:
            pass
