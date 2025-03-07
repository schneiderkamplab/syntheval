# Description: Implementation of auroc metric and plot of roc curves
# Author: Anton D. Lautrup
# Date: 07-11-2023

import numpy as np
import pandas as pd

from syntheval.metrics.core.metric import MetricClass
from syntheval.utils.plot_metrics import plot_roc_curves

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

    def evaluate(self, model = 'log_reg', num_boots = 1) -> float | dict:
        """ Metric that calculates the AUROC difference between a Random Forest model trained on 
        real data and one trained on fake data. Also plots the ROC curves if verbose
        
        Args:
            model (str): 'log_reg' or 'rf_cls'
            num_boots (int): Number of bootstraps runs of the model
        
        Returns:
            dict: AUROC difference between the two models

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> hout = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> AUROC = MetricClassName(real, fake, hout, analysis_target='label', verbose=False, do_preprocessing=False)
            >>> AUROC.evaluate(model='log_reg', num_boots=1)
            {'model': 'log_reg', 'auroc_diff': 0.0}
        """
        try:
            assert self.analysis_target is not None
            assert len(pd.unique(self.real_data[self.analysis_target])) == 2
            assert len(pd.unique(self.synt_data[self.analysis_target])) == 2
            assert self.hout_data is not None
        except AssertionError:
            print(" Warning: AUROC metric did not run, analysis target variable did not have appropriate number levels (i.e. 2) or test data was not supplied!")
            pass
        else:
            real_x, real_y = self.real_data.drop([self.analysis_target], axis=1), self.real_data[self.analysis_target]
            fake_x, fake_y = self.synt_data.drop([self.analysis_target], axis=1), self.synt_data[self.analysis_target]

            hout_x, hout_y = self.hout_data.drop([self.analysis_target], axis=1), self.hout_data[self.analysis_target]
            if model == 'rf_cls':
                model1 = RandomForestClassifier(random_state=42)
                model2 = RandomForestClassifier(random_state=42)
            elif model == 'log_reg':
                model1 = LogisticRegression(random_state=42, max_iter=100)
                model2 = LogisticRegression(random_state=42, max_iter=100)
            else:
                print(f" Error: Unrecognised model name '{model}'!")
                pass

            roc_curves_real = []
            roc_curves_fake = []
            for i in range(num_boots):
                if num_boots != 1:
                    real_x_sub, real_y_sub = resample(real_x, real_y, n_samples=int(len(real_x)/2), stratify=real_y,random_state=i)
                    fake_x_sub, fake_y_sub = resample(fake_x, fake_y, n_samples=int(len(fake_x)/2), stratify=fake_y, random_state=i)
                else:
                    real_x_sub, real_y_sub = real_x, real_y
                    fake_x_sub, fake_y_sub = fake_x, fake_y
                
                model1.fit(real_x_sub, real_y_sub)
                model2.fit(fake_x_sub, fake_y_sub)
                y1_probs = model1.predict_proba(hout_x)[:, 1]
                y2_probs = model2.predict_proba(hout_x)[:, 1]
                
                # Calculate ROC curve for the subsampled model
                fpr1, tpr1, _ = roc_curve(hout_y, y1_probs)
                fpr2, tpr2, _ = roc_curve(hout_y, y2_probs)

                roc_curves_real.append((fpr1, tpr1))
                roc_curves_fake.append((fpr2, tpr2))
            
            mean_fpr = np.linspace(0, 1, len(fpr1))

            tprs_real, tprs_fake = [], []

            for fpr, tpr in roc_curves_real:
                tprs_real.append(np.interp(mean_fpr, fpr, tpr))

            mean_tpr_real = np.mean(tprs_real, axis=0)
            std_tpr_real = np.std(tprs_real, axis=0)

            for fpr, tpr in roc_curves_fake:
                tprs_fake.append(np.interp(mean_fpr, fpr, tpr))

            mean_tpr_fake = np.mean(tprs_fake, axis=0)
            std_tpr_fake = np.std(tprs_fake, axis=0)

            # Calculate AUROC for the mean ROC curve
            roc_auc_mean_real = auc(mean_fpr, mean_tpr_real)
            roc_auc_mean_fake = auc(mean_fpr, mean_tpr_fake)

            if self.verbose: plot_roc_curves([mean_fpr, mean_tpr_real, roc_auc_mean_real], 
                                             [mean_fpr, mean_tpr_real, std_tpr_real], 
                                             [mean_fpr, mean_tpr_fake, roc_auc_mean_fake],
                                             [mean_fpr, mean_tpr_fake, std_tpr_fake],
                                             model, 'roc_curves')

            self.results = {'model': model, 'auroc_diff': roc_auc_mean_fake - roc_auc_mean_real}
            return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        try:
            assert self.analysis_target is not None
            assert len(pd.unique(self.real_data[self.analysis_target])) == 2
            assert len(pd.unique(self.synt_data[self.analysis_target])) == 2
            assert self.hout_data is not None
        except AssertionError:
            pass
        else:
            string = """\
| prediction AUROC difference (%7s)    :   %.4f           |""" % (self.results['model'], self.results['auroc_diff'])
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
            return [{'metric': 'auroc', 'dim': 'u', 
                     'val': self.results['auroc_diff'], 
                     'n_val': np.tanh(2*self.results['auroc_diff']+1), 
                     }]
        else: pass