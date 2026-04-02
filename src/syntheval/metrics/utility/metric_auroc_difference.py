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

class PredictionAUROCDifference(MetricClass):
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

    def evaluate(self, model = 'log_reg', num_boots = 1, full_output: bool = False) -> float | dict:
        """ Metric that calculates the AUROC difference between a Random Forest model trained on 
        real data and one trained on fake data. Also plots the ROC curves if verbose
        
        Args:
            model (str): 'log_reg' or 'rf_cls'
            num_boots (int): Number of bootstraps runs of the model
            full_output (bool): whether to return the full results dictionary or just the auroc_diff value
        
        Returns:
            dict: AUROC difference between the two models

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> hout = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'label': [0, 1, 0]})
            >>> AUROC = PredictionAUROCDifference(real, fake, hout, analysis_target='label',
            ...     verbose=False, do_preprocessing=False, plot_figures=False)
            >>> AUROC.evaluate(model='log_reg', num_boots=1) # doctest: +ELLIPSIS
            {'model': 'log_reg', 'auroc results': ..., 'auroc_diff': 0.0}
        """
        try:
            assert self.analysis_target is not None, "SynthEval(auroc): metric did not run, no analysis target variable(s) supplied!"

            target_vars = [
                key for (key, value) in self.analysis_target.target_types.items() 
                if isinstance(value, int) and value == 2
                ]
            
            assert target_vars != [], "SynthEval(auroc): metric did not run, no categorical target variables with exactly 2 unique values!"
            assert any(len(pd.unique(self.synt_data[var])) == 2 for var in target_vars), "SynthEval(auroc): metric did not run, synthetic data is monotonic in all the target variable(s)!"
            assert self.hout_data is not None, "SynthEval(auroc): metric did not run, no holdout data supplied!"
            assert model in ['rf_cls', 'log_reg'], "SynthEval(auroc): metric did not run, unrecognised model name supplied! Use 'rf_cls' or 'log_reg'."
        except AssertionError as e:
            raise AssertionError(e)
        else:
            self.full_output = full_output
            result_rows = []
            for target_var in target_vars:
                # Drop confounder variables for the current target variable (if any)
                confounders = self.analysis_target.confounder_vars[target_var]
                real_data = self.real_data.drop(confounders, axis=1)
                synt_data = self.synt_data.drop(confounders, axis=1)
                hout_data = self.hout_data.drop(confounders, axis=1)

                real_x, real_y = real_data.drop([target_var], axis=1), real_data[target_var]
                fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]
                hout_x, hout_y = hout_data.drop([target_var], axis=1), hout_data[target_var]
                target_var = target_var.replace(' ', '_').lower()

                match model:
                    case 'rf_cls':
                        model1 = RandomForestClassifier(random_state=42)
                        model2 = RandomForestClassifier(random_state=42)
                    case 'log_reg':
                        model1 = LogisticRegression(random_state=42, max_iter=100)
                        model2 = LogisticRegression(random_state=42, max_iter=100)

                roc_curves_real = []
                roc_curves_fake = []
                for i in range(num_boots):
                    if num_boots != 1:
                        real_x_sub, real_y_sub = resample(real_x, real_y, n_samples=int(len(real_x)/2), stratify=real_y, random_state=i)
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

                if self.plot_figures: plot_roc_curves([mean_fpr, mean_tpr_real, roc_auc_mean_real], 
                                                [mean_fpr, mean_tpr_real, std_tpr_real], 
                                                [mean_fpr, mean_tpr_fake, roc_auc_mean_fake],
                                                [mean_fpr, mean_tpr_fake, std_tpr_fake],
                                                f"{model}, predicting {target_var}", 'roc_curves_'+target_var)
                
                result_rows.append({
                    'target_var': target_var,
                    'model': model,
                    'auroc_diff': float(roc_auc_mean_fake - roc_auc_mean_real)
                })

            self.results['model'] = model

            columns = ['target_var', 'model', 'auroc_diff']
            self.results['auroc results'] = pd.DataFrame.from_records(result_rows, columns=columns)

            self.results['auroc_diff'] = float(self.results['auroc results']['auroc_diff'].mean())
            if len(self.results['auroc results']) > 1:
                self.results['auroc_diff_err'] = float(self.results['auroc results']['auroc_diff'].sem()) 
            # self.results = {'model': model, 'auroc_diff': float(roc_auc_mean_fake - roc_auc_mean_real)}
            return self.results
        
    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        row = ("utility", f"Prediction AUROC difference ({self.results['model']:<7})", self.results['auroc_diff'], self.results.get('auroc_diff_err'))
        return [row]

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        rows = []
        if self.results != {}:
            output = [{'metric': 'auroc', 'dim': 'u', 
                     'val': self.results['auroc_diff'],
                     'err': self.results.get('auroc_diff_err'),
                     'n_val': np.tanh(2*self.results['auroc_diff']+1), 
                     'n_err': self.results.get('auroc_diff_err')
                     }]
            if self.full_output and len(self.results['auroc results']) > 1:
                for index, row in self.results['auroc results'].iterrows():
                    output.append({'metric': 'auroc_'+row['target_var'], 'dim': 'u', 
                                'val': row['auroc_diff'],
                                'n_val': np.tanh(2*row['auroc_diff']+1), 
                                })
            return output
        else: pass