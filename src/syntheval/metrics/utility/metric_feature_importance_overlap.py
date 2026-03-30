# Description: Metric for comparing feature importances
# Author: Anton D. Lautrup
# Date: 30-03-2026

import pandas as pd

from typing import List, Literal

from syntheval.metrics.core.metric import MetricClass
from syntheval.utils.plot_metrics import plot_feature_importance_comparison

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class FeatureImportanceOverlap(MetricClass):
    """ Feature importance overlap metric. This metric compares the feature importances of a model trained 
    on real data and a model trained on synthetic data. The metric is based on the number of overlapping 
    features in the top 5, 10, 25 and 50% most important features of the two models.

    Attributes:
    self.real_data : DataFrame
    self.synt_data : DataFrame
    self.hout_data : DataFrame
    self.cat_cols  : list of strings
    self.num_cols  : list of strings

    self.nn_dist   : string keyword
    self.analysis_target: variable name

    self.verbose   : bool (for supressing prints)
    self.plot_figures: bool (for supressing plots)

    """

    def name() -> str:
        """ Name/keyword to reference the metric"""
        return 'fio'

    def type() -> str:
        """ Set to 'privacy', 'utility' or 'fairness' """
        return 'utility'

    def evaluate(self, model: Literal['rf_cls', 'dt_cls', 'log_reg'] = 'rf_cls') -> float | dict:
        """ Metric that calculates the feature importance overlap between a model trained 
        on real data and one trained on fake data. Also plots the feature importances if plot_figures is True.
        
        Args:
            model (str): 'rf_cls', 'dt_cls' or 'log_reg'

        Returns:
            dict: result variables for the metric

        Example:
        >>> import pandas as pd
        >>> real = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4, 5, 6, 4], 'c': [7, 8, 9, 7], 'd': [1, 0, 1, 0], 'target': [0, 1, 0, 1]})
        >>> fake = pd.DataFrame({'a': [1, 2, 3, 1], 'b': [4, 5, 6, 1], 'c': [7, 8, 9, 1], 'd': [1, 0, 1, 0], 'target': [0, 1, 0, 1]})
        >>> FIO = FeatureImportanceOverlap(real, fake, analysis_target='target', plot_figures=False, do_preprocessing=False)
        >>> results = FIO.evaluate(model='rf_cls')
        """
        try:
            assert self.analysis_target is not None, "FIO metric did not run, no analysis target variable supplied!"
            assert len(self.real_data.columns.tolist()) > 3, "FIO metric did not run, the data must have more than 3 columns!"
            assert model in ['rf_cls', 'log_reg', 'dt_cls'], "FIO metric did not run, model must be one of 'rf_cls', 'log_reg' or 'dt_cls'"
        except AssertionError as e:
            raise AssertionError(e)
        else:
            real_x, real_y = self.real_data.drop([self.analysis_target], axis=1), self.real_data[self.analysis_target]
            fake_x, fake_y = self.synt_data.drop([self.analysis_target], axis=1), self.synt_data[self.analysis_target]

            match model:
                case 'rf_cls':
                    model_real = RandomForestClassifier(random_state=42)
                    model_fake = RandomForestClassifier(random_state=42)
                case 'log_reg':
                    model_real = LogisticRegression(random_state=42, max_iter=100)
                    model_fake = LogisticRegression(random_state=42, max_iter=100)
                case 'dt_cls':
                    model_real = DecisionTreeClassifier(random_state=42)
                    model_fake = DecisionTreeClassifier(random_state=42)

            model_real.fit(real_x, real_y); model_fake.fit(fake_x, fake_y)

            importances_real = model_real.feature_importances_
            importances_real = pd.Series(importances_real, index=real_x.columns).sort_values(ascending=False)
            importances_fake = model_fake.feature_importances_
            importances_fake = pd.Series(importances_fake, index=fake_x.columns).sort_values(ascending=False)

            compare_percentages = [0.05, 0.1, 0.25, 0.5]
            for p in compare_percentages:
                top_real = set(importances_real.index[:int(len(importances_real)*p)])
                top_fake = set(importances_fake.index[:int(len(importances_fake)*p)])

                if len(top_real) < 2 or len(top_real) > 100:
                    continue # skip if too few or too many features for meaningful comparison

                overlap = len(top_real.intersection(top_fake)) / len(top_real)
                self.results[f'overlap_top_{int(p*100)}%'] = overlap

            if self.plot_figures:
                # Plot only up to the 20 most important features for readability
                plot_importances_real = importances_real[:20]
                plot_importances_fake = importances_fake[importances_fake.index.isin(plot_importances_real.index)]

                plot_importances_real_names = plot_importances_real.index.tolist()
                plot_feature_importance_comparison(plot_importances_real_names,
                                                        plot_importances_real.values,
                                                        plot_importances_fake.values,
                                                        title=f"{model}, top {min(20, len(importances_real))} features",
                                                        file_name='feature_importances')
        
        return self.results

    def format_output(self) -> List[tuple]:
        """ Return a list of tuples for printing results to the rich console."""
        rows = []
        if self.results.get('overlap_top_5%') is not None:
            rows.append(('utility', f"Feature importance overlap top 5%", self.results.get('overlap_top_5%'), None))
        if self.results.get('overlap_top_10%') is not None:
            rows.append(('utility', f"Feature importance overlap top 10%", self.results.get('overlap_top_10%'), None))
        if self.results.get('overlap_top_25%') is not None:
            rows.append(('utility', f"Feature importance overlap top 25%", self.results.get('overlap_top_25%'), None))
        if self.results.get('overlap_top_50%') is not None:
            rows.append(('utility', f"Feature importance overlap top 50%", self.results.get('overlap_top_50%'), None))
        return rows

    def normalize_output(self) -> List[dict]:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        
        if self.results != {}:
            dict_lst = []
            dict_keys = ['overlap_top_5%', 'overlap_top_10%', 'overlap_top_25%', 'overlap_top_50%']
            
            for key in dict_keys:
                if key not in self.results:
                    self.results[key] = None # ensure all keys are present for consistent output format
                else:
                    dict_lst.append({'metric': f"fio_{key.split('_')[-1]}", 'dim': 'u', 'val': self.results[key], 'n_val': self.results[key]})
            return dict_lst
        else: pass
