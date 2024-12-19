# Description: Epsilon identifiability metric
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np

from syntheval.metrics.core.metric import MetricClass

from scipy.stats import entropy

from syntheval.utils.nn_distance import _knn_distance

def _column_entropy(labels):
        """ Compute the entropy of a column of data
        Args:
            labels (np.array): A column of data

        Returns:
            float: The entropy of the column
        
        Example:
            >>> import numpy as np
            >>> ent = _column_entropy(np.array([1, 1, 2, 2, 3, 3]))
            >>> isinstance(ent, float)
            True
        """
        value, counts = np.unique(np.round(labels), return_counts=True)
        return entropy(counts)

class EpsilonIdentifiability(MetricClass):
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

    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'eps_risk'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'privacy'

    def evaluate(self) -> float | dict:
        """Function for computing the epsilon identifiability risk

        Adapted from:
        Yoon, J., Drumright, L. N., & van der Schaar, M. (2020). Anonymization Through Data Synthesis Using Generative Adversarial Networks (ADS-GAN). 
        IEEE Journal of Biomedical and Health Informatics, 24(8), 2378–2388. [doi:10.1109/JBHI.2020.2980262]

        Returns:
            dict: The results of the metric

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> EI = EpsilonIdentifiability(real, fake, num_cols=['a','b'], nn_dist='euclid', do_preprocessing=False)
            >>> EI.evaluate()
            {'eps_risk': 0.0}
        """

        if self.nn_dist == 'euclid':
            self.real_data = self.real_data[self.num_cols]
            self.synt_data = self.synt_data[self.num_cols] 

        real = np.asarray(self.real_data)

        no, x_dim = np.shape(real)
        W = [_column_entropy(real[:, i]) for i in range(x_dim)]
        W_adjust = 1/(np.array(W)+1e-16)

        in_dists = _knn_distance(self.real_data,self.real_data,self.cat_cols,1,self.nn_dist,W_adjust)[0]
        ext_distances = _knn_distance(self.real_data,self.synt_data,self.cat_cols,1,self.nn_dist,W_adjust)[0]

        R_Diff = ext_distances - in_dists
        identifiability_value = np.sum(R_Diff < 0) / float(no)

        self.results['eps_risk'] = identifiability_value

        if self.hout_data is not None:
            in_dists = _knn_distance(self.hout_data,self.hout_data,self.cat_cols,1,self.nn_dist,W_adjust)[0]
            ext_distances = _knn_distance(self.hout_data,self.synt_data,self.cat_cols,1,self.nn_dist,W_adjust)[0]

            R_Diff = ext_distances - in_dists
            identifiability_value = np.sum(R_Diff < 0) / float(no)

            self.results['priv_loss'] = self.results['eps_risk'] - identifiability_value

        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Epsilon identifiability risk             :   %.4f           |""" % (self.results['eps_risk'])
        if (self.results != {} and self.hout_data is not None):
             string += """       
| Privacy loss (diff. in eps. risk)        :   %.4f           |""" % (self.results['priv_loss'])
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
            output =  [{'metric': 'eps_identif_risk', 'dim': 'p', 
                     'val': self.results['eps_risk'], 
                     'n_val': 1-self.results['eps_risk'], 
                     }]
            if self.hout_data is not None:
                output.extend([{'metric': 'priv_loss_eps', 'dim': 'p', 
                        'val': self.results['priv_loss'], 
                        'n_val': 1-abs(self.results['priv_loss']), 
                        }])
            return output
        else: pass
