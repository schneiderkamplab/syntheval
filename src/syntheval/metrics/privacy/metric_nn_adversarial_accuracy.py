# Description: Nearest neighbour adversarial accuracy implementation
# Author: Anton D. Lautrup
# Date: 21-08-2023

import numpy as np

from syntheval.metrics.core.metric import MetricClass

from syntheval.utils.nn_distance import _knn_distance

def _adversarial_score(real, fake, cat_cols, metric):
    """Function for calculating adversarial score
    
    Args:
        real (DataFrame) : Real dataset
        fake (DataFrame) : Synthetic dataset
        cat_cols (List[str]) : list of strings
        metric (str) : keyword literal for NN module

    Returns:
        float : Adversarial score
    
    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> _adversarial_score(real, fake, [], 'euclid')
        0.0
    """
    left = np.mean(_knn_distance(real, fake, cat_cols, 1, metric)[0] > _knn_distance(real, real, cat_cols, 1, metric)[0])
    right = np.mean(_knn_distance(fake, real, cat_cols, 1, metric)[0] > _knn_distance(fake, fake, cat_cols, 1, metric)[0])
    return 0.5 * (left + right)

def evaluate_dataset_nnaa(real, fake, num_cols, cat_cols, metric, n_resample):
    """Helper function for running adversarial score multiple times if the 
    datasets have much different sizes.
    
    Args:
        real (DataFrame) : Real dataset
        fake (DataFrame) : Synthetic dataset
        num_cols (List[str]) : list of strings
        cat_cols (List[str]) : list of strings
        metric (str) : keyword literal for NN module
        n_resample (int) : number of resample rounds to run if datasets are of different sizes
    
    Returns:
        float, float: Average adversarial score and standard error
    
    Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> evaluate_dataset_nnaa(real, fake, [], [], 'euclid', 1)
        (0.0, 0.0)
    """

    real_fake = len(real)/len(fake)
    fake_real = len(fake)/len(real)

    if any([real_fake >= 2, fake_real >= 2]):
        aa_lst = []
        for _ in range(n_resample):
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
        """Implementation heavily inspired by original paper
        
        Args:
            n_resample (int) : number of resample rounds to run if datasets are of different sizes
        
        Returns:
            dict: Average adversarial score and standard error
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> N = NearestNeighbourAdversarialAccuracy(real, fake, cat_cols=[], num_cols=[], nn_dist='euclid', do_preprocessing=False)
            >>> N.evaluate(n_resample = 1)
            {'avg': 0.0, 'err': 0.0}
        """

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
                        }]
            if self.hout_data is not None:
                output.extend([{'metric': 'priv_loss_nnaa', 'dim': 'p', 
                        'val': self.results['priv_loss'], 
                        'err': self.results['priv_loss_err'], 
                        'n_val': 1-abs(self.results['priv_loss']), 
                        'n_err': self.results['priv_loss_err'], 
                        }])
            return output
        else: pass

    def extra_formatted_output(self) -> dict:
        """Bit for printing the privacy loss together with the other privacy metrics"""
        if (self.results != {} and self.hout_data is not None):
            string = """\
| Privacy loss (diff. in NNAA)             :   %.4f  %.4f   |""" % (self.results['priv_loss'], self.results['priv_loss_err'])
            return {'privacy': string}
        else: pass
