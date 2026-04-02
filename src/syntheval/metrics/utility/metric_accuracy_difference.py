# Description: Metric implementation of the classification accuracy difference.
# Author: Anton D. Lautrup
# Date: 05-03-2023

import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from typing import Literal
from syntheval.metrics.core.metric import MetricClass

model_name_dict = {
    'dt': 'DecisionTreeClassifier',
    'svm': 'SupportVectorMachine',
    'rf': 'RandomForestClassifier', 
    'adaboost': 'AdaBoostClassifier', 
    'logreg': 'LogisticRegression'
    }

def _get_model(model_name: str) -> object:
    """Function for returning a classification model based on the input string"""
    match model_name:
        case 'dt':
            return DecisionTreeClassifier(max_depth=15, random_state=42)
        case 'rf':
            return RandomForestClassifier(n_estimators=10, max_depth=15, random_state=42)
        case 'svm':
            return SVC(kernel='rbf', random_state=42)
        case 'adaboost':
            return AdaBoostClassifier(n_estimators=10, learning_rate=1, random_state=42)
        case 'logreg':
            return LogisticRegression(solver='saga', max_iter=5000, random_state=42)
        case _:
            raise ValueError(f"SynthEval(cls_acc): Model {model_name} not currently implemented!")


def _propagated_err(err_values: pd.Series) -> float:
    """Propagate independent errors for a mean estimate."""
    values = np.asarray(err_values, dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.nansum(values ** 2)) / values.size)


def _series_sem(values: pd.Series) -> float:
    """Return SEM using sample std (ddof=1), mirroring pandas semantics."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.nan
    return float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size))

def class_test(real_models, fake_models, real, fake, test, F1_type):
    """Function for running a training session and getting predictions 
    on the SciPy model provided, and data.
    
    Args:
        real_models (list): List of SciPy models
        fake_models (list): List of SciPy models
        real (list): List of real data
        fake (list): List of synthetic data
        test (list): List of test data
        F1_type (str): Type of F1 score to compute

    Returns:
        np.array: F1 scores for real and fake data
    
    Example:
        >>> import numpy as np
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> real_models = [DecisionTreeClassifier(), DecisionTreeClassifier()]
        >>> fake_models = [DecisionTreeClassifier(), DecisionTreeClassifier()]
        >>> real = [np.array([[1, 2], [3, 4]]) , np.array([0, 1])]
        >>> fake = [np.array([[1, 2], [3, 4]]) , np.array([0, 1])]
        >>> test = [np.array([[1, 2], [3, 4]]) , np.array([0, 1])]

        >>> class_test(real_models, fake_models, real, fake, test, 'weighted') # doctest: +ELLIPSIS
        array([[1., 1.],...])
    """
    res = []
    for r_mod_, f_mod_ in zip(real_models, fake_models):
        r_mod, f_mod = copy.copy(r_mod_), copy.copy(f_mod_)
        r_mod.fit(real[0],real[1])
        f_mod.fit(fake[0],fake[1])

        pred_real = r_mod.predict(test[0])
        pred_fake = f_mod.predict(test[0])

        f1_real = f1_score(test[1],pred_real,average=F1_type)
        f1_fake = f1_score(test[1],pred_fake,average=F1_type)

        res.append([f1_real, f1_fake])
    return np.array(res).T

class ClassificationAccuracy(MetricClass):
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
        return 'cls_acc'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self, cls_models = ['rf', 'adaboost', 'svm', 'logreg'], 
                 F1_type: Literal['micro', 'macro', 'weighted'] = 'weighted', 
                 k_folds: int = 5, 
                 full_output: bool = False,
                 ) -> dict:

        """ Function for evaluating the metric

        Args:
            cls_models (list): List of classification models to use
                - 'dt' : Decision Tree Classifier
                - 'rf' : Random Forest Classifier
                - 'adaboost' : AdaBoost Classifier
                - 'svm' : Support Vector Machine Classifier
                - 'logreg' : Logistic Regression Classifier
            F1_type (str): Type of F1 score to use
                - 'micro' : Calculate metrics globally by counting the total true positives, false negatives and false positives.
                - 'macro' : Calculate metrics for each label, and find their unweighted mean. I.e, emphasize the importance of rare labels.
                - 'weighted' : Calculate metrics for each label, and find their average weighted by support.
            k_folds (int): Number of folds to use in cross-validation

        Returns:
            dict: result variables for the metric

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4, 5, 6, 4], 'label': [1, 0, 1, 0]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3, 1], 'b': [4, 5, 6, 1], 'label': [1, 0, 1, 0]})
            >>> cls_acc = ClassificationAccuracy(real, fake, cat_cols=['label'], analysis_target='label', do_preprocessing=False)
            >>> cls_acc.evaluate(cls_models = ['rf'], k_folds=2) # doctest: +ELLIPSIS
            {'train results': ...
        """
        if self.analysis_target is None:
            raise AssertionError("SynthEval(cls_acc): Analysis target variable(s) not set!")
        
        target_vars = [
            key for (key, value) in self.analysis_target.target_types.items() 
            if isinstance(value, int) and value >= 2
            ]

        if target_vars == []:
            raise AssertionError("SynthEval(cls_acc): No categorical target variables with 2 or more unique values!")
        if cls_models == []:
            raise AssertionError("SynthEval(cls_acc): No classification models provided!")

        train_rows, test_rows = [], []

        self.k_folds = k_folds
        self.models = cls_models
        self.full_output = full_output

        for target_var in target_vars:
            # Drop confounder variables for the current target variable (if any)
            confounders = self.analysis_target.confounder_vars[target_var]
            real_data = self.real_data.drop(confounders, axis=1)
            synt_data = self.synt_data.drop(confounders, axis=1)

            real_x, real_y = real_data.drop([target_var], axis=1), real_data[target_var]
            fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]

            if self.hout_data is not None:
                hout_data = self.hout_data.drop(confounders, axis=1)

                hout_x, hout_y = hout_data.drop([target_var], axis=1), hout_data[target_var]

            real_models = [_get_model(model_name) for model_name in cls_models]
            fake_models = [_get_model(model_name) for model_name in cls_models]
            target_var = target_var.replace(' ', '_').lower()

            res = []
            max_len = max(len(real_y), len(fake_y))
            real_x_sub, real_y_sub = resample(real_x, real_y, n_samples=max_len, stratify=real_y, random_state=42)
            fake_x_sub, fake_y_sub = resample(fake_x, fake_y, n_samples=max_len, stratify=fake_y, random_state=42)

            kf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
            split_iter = zip(kf.split(real_x_sub, real_y_sub), kf.split(fake_x_sub, fake_y_sub))

            for (train_index_real, test_index_real), (train_index_fake, _) in tqdm(
                split_iter, desc='cls_acc', total=k_folds, disable=not self.verbose,
                ):
                real_x_train, real_y_train = real_x_sub.iloc[train_index_real], real_y_sub.iloc[train_index_real]
                real_x_test, real_y_test = real_x_sub.iloc[test_index_real], real_y_sub.iloc[test_index_real]
                fake_x_train, fake_y_train = fake_x_sub.iloc[train_index_fake], fake_y_sub.iloc[train_index_fake]

                res.append(
                    class_test(
                        real_models,
                        fake_models,
                        [real_x_train, real_y_train],
                        [fake_x_train, fake_y_train],
                        [real_x_test, real_y_test],
                        F1_type,
                    )
                )

            class_avg = np.mean(res, axis=0)
            class_err = np.std(res, axis=0, ddof=1) / np.sqrt(k_folds) if k_folds > 1 else np.zeros_like(class_avg)
            class_diff = class_avg[1, :] - class_avg[0, :]
            class_diff_err = np.sqrt(class_err[0, :] ** 2 + class_err[1, :] ** 2)

            for i, model in enumerate(cls_models):
                train_rows.append({
                    'target_var': target_var,
                    'model': model,
                    'TRTR_acc': class_avg[0, i],
                    'TRTR_err': class_err[0, i],
                    'TSTR_acc': class_avg[1, i],
                    'TSTR_err': class_err[1, i],
                    'acc_diff': class_diff[i],
                    'acc_diff_err': class_diff_err[i],
                })

            if self.hout_data is not None:
                holdout_res = class_test(
                    real_models,
                    fake_models,
                    [real_x, real_y],
                    [fake_x, fake_y],
                    [hout_x, hout_y],
                    F1_type,
                )
                for i, model in enumerate(cls_models):
                    test_rows.append({
                        'target_var': target_var,
                        'model': model,
                        'TRTR_acc': holdout_res[0, i],
                        'TSTR_acc': holdout_res[1, i],
                        'acc_diff': holdout_res[1, i] - holdout_res[0, i],
                        'acc_diff_err': np.nan,
                    })

        train_cols = ['target_var', 'model', 'TRTR_acc', 'TRTR_err', 'TSTR_acc', 'TSTR_err', 'acc_diff', 'acc_diff_err']
        test_cols = ['target_var', 'model', 'TRTR_acc', 'TSTR_acc', 'acc_diff', 'acc_diff_err']
        results_df_train = pd.DataFrame.from_records(train_rows, columns=train_cols)
        results_df_test = pd.DataFrame.from_records(test_rows, columns=test_cols)

        self.results['train results'] = results_df_train
        self.results['test results'] = results_df_test

        self.results['avg diff'] = float(results_df_train['acc_diff'].mean())
        self.results['avg diff err'] = _propagated_err(results_df_train['acc_diff_err'])

        if len(results_df_test) > 0:
            self.results['avg diff hout'] = float(results_df_test['acc_diff'].mean())
            self.results['avg diff err hout'] = _series_sem(results_df_test['acc_diff'])
        return self.results
    
    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        if self.results !={}:
            multiple_targets_flag = len(self.results['train results']['target_var'].unique()) > 1
            rows = [('prediction', 'Accuracy Diff. (%d-fold cross val.)' % (self.k_folds), "", "")]
            train_results = self.results['train results']
            if not multiple_targets_flag:
                for model in self.models:
                    model_rows = train_results[train_results['model'] == model]
                    if len(model_rows) == 0:
                        continue

                    trtr_avg = float(model_rows['TRTR_acc'].mean())
                    tstr_avg = float(model_rows['TSTR_acc'].mean())
                    diff_avg = float(model_rows['acc_diff'].mean())
                    diff_err = _propagated_err(model_rows['acc_diff_err'])

                    rows.append((
                        'prediction',
                        f"{model_name_dict.get(model, model):<22} | RR {trtr_avg:.2f} | FR {tstr_avg:.2f}",
                        diff_avg,
                        diff_err,
                    ))

                if len(train_results) > 1:
                    rows.append((
                        'prediction',
                        f"{'Averages':<22} | RR {train_results['TRTR_acc'].mean():.2f} | FR {train_results['TSTR_acc'].mean():.2f}",
                        self.results['avg diff'],
                        self.results['avg diff err'],
                    ))
                
            else:
                target_averages = train_results.groupby('target_var').agg({
                    'TRTR_acc': 'mean',
                    'TSTR_acc': 'mean',
                    'acc_diff': 'mean'
                }).reset_index()

                for _, target_row in target_averages.iterrows():
                    target_rows = train_results[train_results['target_var'] == target_row['target_var']]
                    target_err = _propagated_err(target_rows['acc_diff_err'])

                    rows.append((
                        'prediction',
                        f"Avg. for {target_row['target_var']:<13} | RR {target_row['TRTR_acc']:.2f} | FR {target_row['TSTR_acc']:.2f}",
                        float(target_row['acc_diff']),
                        target_err,
                    ))

                rows.append((
                    'prediction',
                    f"{'Global averages':<22} | RR {train_results['TRTR_acc'].mean():.2f} | FR {train_results['TSTR_acc'].mean():.2f}",
                    self.results['avg diff'],
                    self.results['avg diff err'],
                ))

            if len(self.results['test results']) > 0:
                test_results = self.results['test results']
                rows.append(('prediction', 'Holdout Data Results', "", ""))

                if not multiple_targets_flag:
                    for model in self.models:
                        model_rows = test_results[test_results['model'] == model]
                        if len(model_rows) == 0:
                            continue

                        trtr_avg = float(model_rows['TRTR_acc'].mean())
                        tstr_avg = float(model_rows['TSTR_acc'].mean())
                        diff_avg = float(model_rows['acc_diff'].mean())

                        rows.append((
                            'prediction',
                            f"{model_name_dict.get(model, model):<22} | RR {trtr_avg:.2f} | FR {tstr_avg:.2f}",
                            diff_avg,
                            None,
                        ))
                    if len(test_results) > 1:
                        rows.append((
                            'prediction',
                            f"{'Averages':<22} | RR {test_results['TRTR_acc'].mean():.2f} | FR {test_results['TSTR_acc'].mean():.2f}",
                            self.results['avg diff hout'],
                            self.results['avg diff err hout']
                        ))
                else:
                    target_test_averages = test_results.groupby('target_var').agg(
                        TRTR_acc=('TRTR_acc', 'mean'),
                        TSTR_acc=('TSTR_acc', 'mean'),
                        acc_diff=('acc_diff', 'mean'),
                        acc_diff_sem=('acc_diff', 'sem'),
                    ).reset_index()

                    for _, target_row in target_test_averages.iterrows():
                        rows.append((
                            'prediction',
                            f"Avg. for {target_row['target_var']:<13} | RR {target_row['TRTR_acc']:.2f} | FR {target_row['TSTR_acc']:.2f}",
                            float(target_row['acc_diff']),
                            float(target_row['acc_diff_sem'])
                        ))

                    rows.append((
                        'prediction',
                        f"{'Global averages':<22} | RR {test_results['TRTR_acc'].mean():.2f} | FR {test_results['TSTR_acc'].mean():.2f}",
                        self.results['avg diff hout'],
                        self.results['avg diff err hout']
                    ))
            return rows

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        
        The required format is:
        metric  val  err  n_val  n_err idx_val idx_err
            name1  0.0  0.0    0.0    0.0    None    None
            name2  0.0  0.0    0.0    0.0    0.0     0.0
        """
        if self.results !={}:
            train_results = self.results['train results']
            test_results = self.results['test results']
            multiple_targets_flag = len(train_results['target_var'].unique()) > 1

            avg_diff = self.results.get('avg diff', float(train_results['acc_diff'].mean()))
            avg_diff_err = self.results.get('avg diff err', _propagated_err(train_results['acc_diff_err']))

            output = [{'metric': 'avg_F1_diff', 'dim': 'u',
                       'val': avg_diff,
                       'err': avg_diff_err,
                       'n_val': 1-abs(avg_diff),
                       'n_err': avg_diff_err,
                       }]

            target_groups_train = [
                (target_var, train_results[train_results['target_var'] == target_var])
                for target_var in train_results['target_var'].unique()
            ] if multiple_targets_flag else [(None, train_results)]

            if self.full_output:
                for target_var, target_rows in target_groups_train:
                    for model in self.models:
                        model_rows = target_rows[target_rows['model'] == model]
                        if len(model_rows) == 0:
                            continue

                        metric_prefix = f'{target_var}_{model}' if target_var is not None else model
                        syn_f1 = float(model_rows['TSTR_acc'].mean())
                        syn_f1_err = _propagated_err(model_rows['TSTR_err'])
                        model_diff = float(model_rows['acc_diff'].mean())
                        model_diff_err = _propagated_err(model_rows['acc_diff_err'])

                        output.extend([{'metric': f'{metric_prefix}_syn_F1', 'dim': 'u',
                            'val': syn_f1,
                            'err': syn_f1_err,
                            'n_val': syn_f1,
                            'n_err': syn_f1_err,
                            }])

                        output.extend([
                            {'metric': f'{metric_prefix}_F1_diff', 'dim': 'u',
                                'val': model_diff,
                                'err': model_diff_err,
                                'n_val': 1-abs(model_diff),
                                'n_err': model_diff_err,
                            }])
            if len(test_results) > 0:
                avg_diff_hout = self.results.get('avg diff hout', float(test_results['acc_diff'].mean()))
                output.extend([{'metric': 'avg_F1_diff_hout', 'dim': 'u',
                       'val': avg_diff_hout,
                       'err': self.results.get('avg diff err hout', None),
                       'n_val': 1-abs(avg_diff_hout),
                       'n_err': self.results.get('avg diff err hout', None),
                       }])

                target_groups_test = [
                    (target_var, test_results[test_results['target_var'] == target_var])
                    for target_var in test_results['target_var'].unique()
                ] if multiple_targets_flag else [(None, test_results)]

                if self.full_output:
                    for target_var, target_rows in target_groups_test:
                        for model in self.models:
                            model_rows = target_rows[target_rows['model'] == model]
                            if len(model_rows) == 0:
                                continue

                            metric_prefix = f'{target_var}_{model}' if target_var is not None else model
                            syn_f1_hout = float(model_rows['TSTR_acc'].mean())
                            model_diff_hout = float(model_rows['acc_diff'].mean())

                            output.extend([{'metric': f'{metric_prefix}_syn_F1_hout', 'dim': 'u',
                                'val': syn_f1_hout,
                                'n_val': syn_f1_hout,
                                }])

                            output.extend([
                                {'metric': f'{metric_prefix}_F1_diff_hout', 'dim': 'u',
                                    'val': model_diff_hout,
                                    'n_val': 1-abs(model_diff_hout),
                                }])
            return output
        else: pass