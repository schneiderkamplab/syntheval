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
    """ Feature ranking metric. This metric compares feature importances and top k features ranking 
    of a model trained on real data and a model trained on synthetic data. The metric is based on the 
    absolute difference in feature importance values and on the number of overlapping features in 
    the top 5, 10, 25 and 50% most important features of the two models.

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
        """ Metric that calculates the feature importance overlap and ranking consistency between a model trained 
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
            assert self.analysis_target is not None, "SynthEval(fio): metric did not run, no analysis target variable(s) supplied!"
            target_vars = [
                key for (key, value) in self.analysis_target.target_types.items() 
                if isinstance(value, int) and value >= 2
                ]
            assert target_vars != [], "SynthEval(fio): No categorical target variables with 2 or more unique values!"
            assert len(self.real_data.columns.tolist()) > 3, "SynthEval(fio): metric did not run, the data must have more than 3 columns!"
            assert model in ['rf_cls', 'log_reg', 'dt_cls'], "SynthEval(fio): metric did not run, model must be one of 'rf_cls', 'log_reg' or 'dt_cls'"
        except AssertionError as e:
            raise AssertionError(e)
        else:
            result_rows = []
            compare_percentages = [0.05, 0.1, 0.25, 0.5]

            for target_var in target_vars:
                # Drop confounder variables for the current target variable (if any)
                confounders = self.analysis_target.confounder_vars[target_var]
                real_data = self.real_data.drop(confounders, axis=1)
                synt_data = self.synt_data.drop(confounders, axis=1)

                real_x, real_y = real_data.drop([target_var], axis=1), real_data[target_var]
                fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]
                target_var = target_var.replace(' ', '_').lower()

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
                
                # Compute ranking consistency for top k features
                res = {}
                for p in compare_percentages:
                    top_real = set(importances_real.index[:int(len(importances_real)*p)])
                    top_fake = set(importances_fake.index[:int(len(importances_fake)*p)])

                    if len(top_real) < 2 or len(top_real) > 100:
                        continue # skip if too few or too many features for meaningful comparison

                    overlap = len(top_real.intersection(top_fake)) / len(top_real)
                    res['top_'+str(int(p*100))+ '%'] = overlap
                
                # Compute importance value consistency (e.g. mean absolute error of importance values)
                importance_diff = (importances_real - importances_fake).abs()
                res['mae_importance'] = importance_diff.mean()
                res['mae_weighted_importance'] = (importance_diff * importances_real).sum() / importances_real.sum()

                result_rows.append({
                    'target_var': target_var,
                    'model': model,
                    **res
                })

                if self.plot_figures:
                    # Plot up to 20 features that are important in either model, ordered by the maximum importance.
                    plot_importances = pd.concat(
                        [importances_real.rename('real'), importances_fake.rename('synthetic')],
                        axis=1,
                    ).fillna(0.0)
                    plot_importances = plot_importances.assign(
                        max_importance=plot_importances.max(axis=1)
                    ).sort_values('max_importance', ascending=False).head(20)

                    plot_importances_real_names = plot_importances.index.tolist()
                    plot_feature_importance_comparison(plot_importances_real_names,
                                                            plot_importances['real'].values,
                                                            plot_importances['synthetic'].values,
                                                            title=f"{model}, predicting {target_var}\ntop {min(20, len(importances_real))} features",
                                                            file_name='feature_importance_'+target_var)
            if result_rows == []:
                self.results = {}
                return self.results
            else:
                self.results['model'] = model

            # Average results across target variables if multiple
            if len(result_rows) > 1:
                df_results = pd.DataFrame(result_rows)
                self.results['fio_diff'] = float(df_results['mae_importance'].mean())
                self.results['fio_diff_weighted'] = float(df_results['mae_weighted_importance'].mean())
                self.results['fio_diff_err'] = float(df_results['mae_importance'].sem())
                self.results['fio_diff_weighted_err'] = float(df_results['mae_weighted_importance'].sem())
                for p in compare_percentages:
                    #check if the column exists (it may not if there were too few features for comparison)
                        if 'top_'+str(int(p*100))+ '%' in df_results.columns:
                            self.results[f'fio_top_{int(p*100)}%'] = float(df_results['top_'+str(int(p*100))+ '%'].mean())
                            self.results[f'fio_top_{int(p*100)}%_err'] = float(df_results['top_'+str(int(p*100))+ '%'].sem())
                        else:
                            continue
            else:
                self.results['fio_diff'] = result_rows[0]['mae_importance']
                self.results['fio_diff_weighted'] = result_rows[0]['mae_weighted_importance']
                self.results = {'fio_' + key: result_rows[0][key] for key in result_rows[0] if key.startswith('top_')}
        return self.results

    def format_output(self) -> List[tuple]:
        """ Return a list of tuples for printing results to the rich console."""
        if self.results !={}:
            rows = [('prediction', f'Feature Importance Overlap ({self.results.get("model"):.7})', "", "")]
            if self.results.get('fio_diff') is not None:
                rows.append(('prediction', f"Importance difference", self.results['fio_diff'], self.results.get('fio_diff_err')))
            if self.results.get('fio_diff_weighted') is not None:
                rows.append(('prediction', f"Weighted importance difference", self.results['fio_diff_weighted'], self.results.get('fio_diff_weighted_err')))
            if self.results.get('fio_top_5%') is not None:
                rows.append(('prediction', f"Ranking consistency top 5%", self.results['fio_top_5%'], self.results.get('fio_top_5%_err')))
            if self.results.get('fio_top_10%') is not None:
                rows.append(('prediction', f"Ranking consistency top 10%", self.results['fio_top_10%'], self.results.get('fio_top_10%_err')))
            if self.results.get('fio_top_25%') is not None:
                rows.append(('prediction', f"Ranking consistency top 25%", self.results['fio_top_25%'], self.results.get('fio_top_25%_err')))
            if self.results.get('fio_top_50%') is not None:
                rows.append(('prediction', f"Ranking consistency top 50%", self.results['fio_top_50%'], self.results.get('fio_top_50%_err')))
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
            dict_keys = ['fio_diff', 'fio_diff_weighted', 'fio_top_5%', 'fio_top_10%', 'fio_top_25%', 'fio_top_50%']
            
            for key in dict_keys:
                if key not in self.results:
                    self.results[key] = None # ensure all keys are present for consistent output format
                elif key.startswith('fio_diff') or key.startswith('fio_diff_weighted'):
                    dict_lst.append({'metric': key, 
                                     'dim': 'u', 
                                     'val': self.results[key], 
                                     'err': self.results.get(f'{key}_err'), 
                                     'n_val': 1-self.results[key], 
                                     'n_err': self.results.get(f'{key}_err')
                                     })
                else:
                    dict_lst.append({'metric': key, 
                                     'dim': 'u', 
                                     'val': self.results[key], 
                                     'err': self.results.get(f'{key}_err'), 
                                     'n_val': self.results[key], 
                                     'n_err': self.results.get(f'{key}_err')
                                     })
            return dict_lst
        else: pass
