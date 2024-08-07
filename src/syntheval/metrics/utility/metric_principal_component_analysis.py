# Description: Principal component analysis plot
# Author: Anton D. Lautrup
# Date: 23-08-2023

import pandas as pd
import numpy as np

from typing import Literal

from ..core.metric import MetricClass

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ...utils.plot_metrics import plot_principal_components, plot_own_principal_component_pairplot
from ...utils.preprocessing import stack

class PrincipalComponentAnalysis(MetricClass):
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
        return 'pca'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, num_components = 2, preprocess: Literal['mean', 'std'] = 'mean') -> float | dict:
        """ Function for evaluating the metric
        
        Args:
        num_components : int
            Number of principal components to plot, the metrics are based on the first two regardless
        preprocess : str
            Preprocessing method for the data, either 'mean' or 'std'
        """

        try:
            assert ((not self.verbose or (self.analysis_target is not None and self.analysis_target in self.cat_cols)) and len(self.num_cols)>=2)
        except AssertionError:
            if len(self.num_cols)<2:
                print('Error: Principal component analysis did not run, too few nummerical attributes!')
            elif self.analysis_target is None: 
                print('Error: Principal component analysis did not run, analysis class variable not set!')
            else:
                print('Error: Principal component analysis did not run, provided class not in list of categoricals!')
            pass
        else:
            if preprocess == 'mean':
                real_data = self.encoder.decode(self.real_data)
                synt_data = self.encoder.decode(self.synt_data)
                r_scaled = real_data[self.num_cols] - real_data[self.num_cols].mean()
                f_scaled = synt_data[self.num_cols] - synt_data[self.num_cols].mean()
            else:
                r_scaled = StandardScaler().fit_transform(self.real_data[self.num_cols])
                f_scaled = StandardScaler().fit_transform(self.synt_data[self.num_cols])

            pca = PCA(n_components=num_components)
            r_pca = pca.fit_transform(r_scaled)
            f_pca = pca.transform(f_scaled)

            labels = [ f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)]

            synt_pca = PCA(n_components=num_components)
            s_pca = synt_pca.fit_transform(f_scaled)

            var_difference = sum(abs(pca.explained_variance_ratio_- synt_pca.explained_variance_ratio_))

            len_r = np.sqrt(pca.components_[0].dot(pca.components_[0]))
            len_f = np.sqrt(synt_pca.components_[0].dot(synt_pca.components_[0]))

            angle_diff = min([np.arccos(pca.components_[0] @ (s*synt_pca.components_[0])) for s in [1,-1]])/(len_r*len_f)

            self.results = {'exp_var_diff': var_difference/2, 'comp_angle_diff': 2*angle_diff/np.pi}
            if len(self.num_cols) > 5 :
                r_sort = np.argsort(abs(pca.components_[0]))[::-1]
                s_sort = np.argsort(abs(synt_pca.components_[0]))[::-1]
                self.results['pca_max_cont_real'] = [self.num_cols[i] for i in r_sort[:5]]
                self.results['pca_max_cont_synt'] = [self.num_cols[i] for i in s_sort[:5]]
            
            r_pca = pd.DataFrame(r_pca,columns=labels)
            f_pca = pd.DataFrame(f_pca,columns=labels)
            s_pca = pd.DataFrame(s_pca,columns=labels)

            if self.verbose: 
                r_pca['target'] = self.real_data[self.analysis_target]
                f_pca['target'] = self.synt_data[self.analysis_target]
                plot_principal_components(r_pca,f_pca)
                plot_own_principal_component_pairplot(stack(r_pca,s_pca))
            return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        if self.results != {}:
            string = """\
| PCA difference in eigenvalues (exp. var.):   %.4f           |
| PCA angle diff. between eigenvectors     :   %.4f           |""" % (self.results['exp_var_diff'], 
                                                                      self.results['comp_angle_diff'])
            return string
        else: pass

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            return [{'metric': 'pca_eigval_diff', 'dim': 'u', 
                     'val': self.results['exp_var_diff'], 
                     'n_val': 1-self.results['exp_var_diff'] 
                     },
                     {'metric': 'pca_eigvec_ang', 'dim': 'u', 
                     'val': self.results['comp_angle_diff'], 
                     'n_val': 1-self.results['comp_angle_diff'] 
                     }]
        else: pass






