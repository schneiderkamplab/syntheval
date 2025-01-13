# Description: Principal component analysis plot
# Author: Anton D. Lautrup
# Date: 23-08-2023

import pandas as pd
import numpy as np

from typing import Literal

from syntheval.metrics.core.metric import MetricClass

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pcametric import PCAMetric

from syntheval.utils.plot_metrics import plot_principal_components, plot_own_principal_component_pairplot
from syntheval.utils.preprocessing import stack

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

    def evaluate(self, num_components = 2, preprocess: Literal['mean', 'std'] = 'mean', use_cats: bool = False) -> float | dict:
        """ Function for evaluating the metric
        
        Args:
            num_components (int): Number of principal components to consider in visualization
            preprocess (str): Preprocessing method for the data (mean or std)
            use_cats (bool): Include categorical variables in the analysis
        
        Returns:
            dict: The difference in eigenvalues and eigenvectors of the principal components of the two

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> PCA = PrincipalComponentAnalysis(real, fake, cat_cols=[], analysis_target='a', do_preprocessing=False, verbose=False)
            >>> PCA.evaluate()
        """

        try:
            assert ((self.analysis_target is not None and self.analysis_target in self.cat_cols) and (use_cats or len(self.num_cols)>=2))
        except AssertionError:
            if self.verbose:
                if use_cats or len(self.num_cols)<2:
                    print('Error: Principal component analysis did not run, too few attributes!')
                elif self.analysis_target is None: 
                    print('Error: Principal component analysis did not run, analysis class variable not set!')
                else:
                    print('Error: Principal component analysis did not run, provided class not in list of categoricals!')
            pass
        else:
            if use_cats:
                select_cols = self.num_cols + self.cat_cols
            else:
                select_cols = self.num_cols

            # Get PCA metrics and projections
            res = PCAMetric(self.real_data[select_cols], self.synt_data[select_cols], preprocess=preprocess)[0]

            self.results = res

            if self.verbose:  # For the pca plots we have to redo some stuff

                if preprocess == 'mean':
                    real_data = self.encoder.decode(self.real_data)
                    synt_data = self.encoder.decode(self.synt_data)

                    if use_cats:
                        real_data[self.cat_cols] = self.real_data[self.cat_cols]
                        synt_data[self.cat_cols] = self.synt_data[self.cat_cols]
                    
                    r_scaled = real_data[select_cols] - real_data[select_cols].mean()
                    f_scaled = synt_data[select_cols] - synt_data[select_cols].mean()
                else:
                    r_scaled = StandardScaler().fit_transform(self.real_data[select_cols])
                    f_scaled = StandardScaler().fit_transform(self.synt_data[select_cols])

                pca = PCA(n_components=num_components)
                r_proj = pca.fit_transform(r_scaled)
                f_proj = pca.transform(f_scaled)

                labels = [ f"PC {i+1} ({var:.1f}%)" for i, var in enumerate(pca.explained_variance_ratio_ * 100)]

                s_pca = PCA(n_components=num_components)
                s_proj = s_pca.fit_transform(f_scaled)

                if len(select_cols) > 5 :
                    r_sort = np.argsort(abs(pca.components_[0]))[::-1]
                    s_sort = np.argsort(abs(s_pca.components_[0]))[::-1]
                    self.results['pca_max_cont_real'] = [select_cols[i] for i in r_sort[:5]]
                    self.results['pca_max_cont_synt'] = [select_cols[i] for i in s_sort[:5]]
            
                r_proj = pd.DataFrame(r_proj,columns=labels)
                f_proj = pd.DataFrame(f_proj,columns=labels)
                s_proj = pd.DataFrame(s_proj,columns=labels)

                r_proj['target'] = self.real_data[self.analysis_target]
                f_proj['target'] = self.synt_data[self.analysis_target]
                plot_principal_components(r_proj,f_proj)
                plot_own_principal_component_pairplot(stack(r_proj,s_proj))

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






