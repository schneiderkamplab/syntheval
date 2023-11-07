# Description: Implementation of auroc metric and plot of roc curves
# Author: Anton D. Lautrup
# Date: 07-11-2023

import numpy as np
import pandas as pd

from ..core.metric import MetricClass
from ...utils.plot_metrics import plot_roc_curves

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

class MetricClassName(MetricClass):
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

    self.verbose   : bool (mainly for supressing prints and plots)

    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'auroc_diff'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, model = 'log_reg', num_subsamples = 1) -> float | dict:
        """ Metric that calculates the AUROC difference between a Random Forest model trained on 
        real data and one trained on fake data. Also plots the ROC curves if verbose
        
        model: 'log_reg' or 'rf_cls'
        
        """
        try:
            assert len(pd.unique(self.real_data[self.analysis_target])) == 2
            assert self.hout_data is not None
        except AssertionError:
            print("Error: AUROC metric did not run, analysis target variable did not have appropriate number levels (2) or test data was not supplied!'")
            pass
        else:
            real_x, real_y = self.real_data.drop([self.analysis_target], axis=1), self.real_data[self.analysis_target]
            fake_x, fake_y = self.synt_data.drop([self.analysis_target], axis=1), self.synt_data[self.analysis_target]

            hout_x, hout_y = self.hout_data.drop([self.analysis_target], axis=1), self.hout_data[self.analysis_target]
            if model == 'rf_cls':
                model1 = RandomForestClassifier(random_state=42)
                model2 = RandomForestClassifier(random_state=42)
            elif model == 'log_reg':
                model1 = LogisticRegression(random_state=42)
                model2 = LogisticRegression(random_state=42)
            else:
                print(f"Error: Unrecognised model name '{model}'!")
                pass
            
            model2.fit(fake_x, fake_y)
            y2_probs = model2.predict_proba(hout_x)[:, 1]

            fpr2, tpr2, _ = roc_curve(hout_y, y2_probs)
            roc_auc2 = auc(fpr2, tpr2)

            roc_curves = []
            for i in range(num_subsamples):
                if num_subsamples != 1:
                    real_x_sub, real_y_sub = resample(real_x, real_y, n_samples=int(len(real_x)/2), random_state=i)
                else:
                    real_x_sub, real_y_sub = real_x, real_y
                
                model1.fit(real_x_sub, real_y_sub)
                y1_probs = model1.predict_proba(hout_x)[:, 1]
                
                # Calculate ROC curve for the subsampled model
                fpr1, tpr1, _ = roc_curve(hout_y, y1_probs)
                roc_curves.append((fpr1, tpr1))
            
            #mean_fpr = np.linspace(0, 1, 100)
            mean_fpr = fpr2
            tprs = []

            for fpr, tpr in roc_curves:
                tprs.append(np.interp(mean_fpr, fpr, tpr))

            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)

            # Calculate AUROC for the mean ROC curve
            roc_auc_mean = auc(mean_fpr, mean_tpr)

            if self.verbose: plot_roc_curves([mean_fpr, mean_tpr, roc_auc_mean], [mean_fpr,mean_tpr, std_tpr], [fpr2, tpr2, roc_auc2], model, 'roc_curves')

            self.results = {'model': model, 'auroc_diff': roc_auc_mean - roc_auc2}
            return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        try:
            assert len(pd.unique(self.real_data[self.analysis_target])) == 2
            assert self.hout_data is not None
        except AssertionError:
            pass
        else:
            string = """\
| prediction AUROC difference (%.7s)    :   %.4f           |""" % (self.results['model'], self.results['auroc_diff'])
            return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best. 
        
        Depending on the metric, you may want to check what scores are actually
        realistically possible and adjust this scale using nonlinearities so that 
        values below 0.95 are actually possible and that the metric does not 
        universally drag the average up. See the existing metrics for examples.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """
        val_non_lin = np.exp(-10*self.results['auroc_diff'])

        return {'val': [val_non_lin], 'err': [0]}
