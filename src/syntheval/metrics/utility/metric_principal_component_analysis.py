# Description: Principal component analysis plot
# Author: Anton D. Lautrup
# Date: 23-08-2023

import pandas as pd

from ..core.metric import MetricClass

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ...utils.plot_metrics import plot_principal_components

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
        return 'plot_pca'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self) -> float | dict:
        """ Function for evaluating the metric"""
        if (self.analysis_target is not None and self.analysis_target in self.cat_cols):
            r_scaled = StandardScaler().fit_transform(self.real_data[self.num_cols])
            f_scaled = StandardScaler().fit_transform(self.synt_data[self.num_cols])

            pca = PCA(n_components=2)
            r_pca = pca.fit_transform(r_scaled)
            f_pca = pca.transform(f_scaled)

            r_pca = pd.DataFrame(r_pca,columns=['PC1', 'PC2'])
            f_pca = pd.DataFrame(f_pca,columns=['PC1', 'PC2'])
            r_pca['target'] = self.real_data[self.analysis_target]
            f_pca['target'] = self.synt_data[self.analysis_target]
            plot_principal_components(r_pca,f_pca)
            pass
        elif self.analysis_target is None: 
            print('Error: Principal component analysis did not run, analysis class variable not set!')
            pass
        else:
            print('Error: Principal component analysis did not run, provided class not in list of categoricals!')
            pass

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        pass

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """
        pass






