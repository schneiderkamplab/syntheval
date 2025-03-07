# Description: Metric implementation of the classification accuracy difference.
# Author: Anton D. Lautrup
# Date: 05-03-2023

import copy

import numpy as np
from tqdm import tqdm

from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

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

        >>> class_test(real_models, fake_models, real, fake, test, 'micro') # doctest: +ELLIPSIS
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

    def evaluate(self, cls_models = ['rf', 'adaboost', 'svm', 'logreg'], F1_type='micro', k_folds=5, full_output=False) -> dict:

        """ Function for evaluating the metric

        Args:
            cls_models (list): List of classification models to use
                - 'dt' : Decision Tree Classifier
                - 'rf' : Random Forest Classifier
                - 'adaboost' : AdaBoost Classifier
                - 'svm' : Support Vector Machine Classifier
                - 'logreg' : Logistic Regression Classifier
            F1_type (str): Type of F1 score to use
            k_folds (int): Number of folds to use in cross-validation

        Returns:
            dict: result variables for the metric

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4, 5, 6, 4], 'label': [1, 0, 1, 0]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3, 1], 'b': [4, 5, 6, 1], 'label': [1, 0, 1, 0]})
            >>> cls_acc = ClassificationAccuracy(real, fake, cat_cols=['label'], analysis_target='label', do_preprocessing=False)
            >>> cls_acc.evaluate(cls_models = ['rf'], k_folds=2 ) # doctest: +ELLIPSIS
            {'rf': ...
        """
        try:
            assert self.analysis_target is not None, "SynthEval(cls_acc): Analysis target variable not set!"
            assert self.analysis_target in self.cat_cols, "SynthEval(cls_acc): Analysis target variable not categorical!"
            assert len(self.synt_data[self.analysis_target].unique()) >= 2, "SynthEval(cls_acc): Synthetic label column has less than 2 unique values!"
            assert cls_models != [], "SynthEval(cls_acc): No classification models provided!"
        except AssertionError as e:
            print(e)
            return {}
        else:
            self.results = {}
            real_x, real_y = self.real_data.drop([self.analysis_target], axis=1), self.real_data[self.analysis_target]
            fake_x, fake_y = self.synt_data.drop([self.analysis_target], axis=1), self.synt_data[self.analysis_target]  
            
            # Get models
            real_models = [_get_model(model_name) for model_name in cls_models]
            fake_models = [_get_model(model_name) for model_name in cls_models]
            self.models = cls_models

            # Setup Validation Run
            res = []
            max_len = len(real_y) if len(real_y)>len(fake_y) else len(fake_y) # Get the maximum length of the two datasets for resample

            real_x_sub, real_y_sub = resample(real_x, real_y, n_samples=max_len, stratify=real_y, random_state=42)
            fake_x_sub, fake_y_sub = resample(fake_x, fake_y, n_samples=max_len, stratify=fake_y, random_state=42)

            kf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)

            for (train_index_real, test_index_fake), (train_index_fake, _) in tqdm(zip(kf.split(real_x_sub, real_y_sub), kf.split(fake_x_sub, fake_y_sub)), 
                                                                                            desc='cls_acc', total=k_folds, disable = not self.verbose):
                
                real_x_train, real_y_train = real_x_sub.iloc[train_index_real], real_y_sub.iloc[train_index_real]
                real_x_test, real_y_test = real_x_sub.iloc[test_index_fake], real_y_sub.iloc[test_index_fake]
                fake_x_train, fake_y_train = fake_x_sub.iloc[train_index_fake], fake_y_sub.iloc[train_index_fake]

                res.append(class_test(real_models,fake_models,[real_x_train, real_y_train],
                                                            [fake_x_train, fake_y_train],
                                                            [real_x_test, real_y_test], F1_type))
                
            self.results = {}
            class_avg = np.mean(res,axis=0)
            class_err = np.std(res,axis=0,ddof=1)/np.sqrt(k_folds)
            class_diff = class_avg[1,:]-class_avg[0,:]
            class_diff_err = np.sqrt(class_err[0,:]**2+class_err[1,:]**2)

            self.full_output = full_output
            self.k_folds = k_folds      # Make these items accessible for the user
            self.class_avg = class_avg

            for i, model in enumerate(cls_models):
                self.results[model] = {'rr_val_acc': class_avg[0,i], 'rr_val_err': class_err[0,i], 'fr_val_acc': class_avg[1,i], 'fr_val_err': class_err[1,i]}

            self.results['avg diff'] = np.mean(class_diff)
            self.results['avg diff err'] = 1/len(cls_models)*np.sqrt(sum(class_diff_err**2))
            
            if (self.hout_data is not None):
                holdout_res = class_test(real_models, fake_models, [real_x, real_y],
                                                                    [fake_x, fake_y],
                                                                    [self.hout_data.drop([self.analysis_target],axis=1), 
                                                                    self.hout_data[self.analysis_target]], F1_type)

                for i, model in enumerate(cls_models):
                    self.results[model]['rr_test_acc'] = holdout_res[0,i]
                    self.results[model]['fr_test_acc'] = holdout_res[1,i]
                
                holdout_diff = holdout_res[1,:]-holdout_res[0,:]

                self.holdout_res = holdout_res
                
                self.results['avg diff hout'] = np.mean(holdout_diff)
                self.results['avg diff err hout'] = np.std(holdout_diff,ddof=1)/len(cls_models)

            return self.results
        
    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        if self.results != {}:
            avgs = self.class_avg
            string = """\
+---------------------------------------------------------------+

Classification accuracy test     
avg. of %d-fold cross val.:
classifier model             acc_rr  acc_fr    diff   error
+---------------------------------------------------------------+
""" % self.k_folds
            for model in self.models:
                mod_dict = self.results[model]
                string += f"""\
| {model_name_dict[model]:<23} :   {mod_dict['rr_val_acc']:.4f}  {mod_dict['fr_val_acc']:.4f}  {mod_dict['fr_val_acc']-mod_dict['rr_val_acc']:>7.4f}  {np.sqrt(mod_dict['rr_val_err']**2+mod_dict['fr_val_err']**2):.4f}   |
"""
            string += f"""\
+---------------------------------------------------------------+
| Average                 :   {np.mean(avgs[0,:]):.4f}  {np.mean(avgs[1,:]):.4f}  {self.results['avg diff']:>7.4f}  {self.results['avg diff err']:.4f}   |
"""
            if (self.hout_data is not None):
                hdiff = self.holdout_res[1,:]-self.holdout_res[0,:]
                string += """\
+---------------------------------------------------------------+

hold out data results:
+---------------------------------------------------------------+
"""
                for i, model in enumerate(self.models):
                    mod_dict = self.results[model]
                    string += f"""\
| {model_name_dict[model]:<23} :   {mod_dict['rr_test_acc']:.4f}  {mod_dict['fr_test_acc']:.4f}  {hdiff[i]:>7.4f}           |
"""
                string += f"""\
+---------------------------------------------------------------+
| Average                 :   {np.mean(self.holdout_res[0,:]):.4f}  {np.mean(self.holdout_res[1,:]):.4f}  {self.results['avg diff hout']:>7.4f}  {self.results['avg diff err hout']:.4f}   |
"""
            return string
        else: pass

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        
        The required format is:
        metric  val  err  n_val  n_err idx_val idx_err
            name1  0.0  0.0    0.0    0.0    None    None
            name2  0.0  0.0    0.0    0.0    0.0     0.0
        """
        if self.results !={}:
            output = [{'metric': 'avg_F1_diff', 'dim': 'u',
                       'val': self.results['avg diff'], 
                       'err': self.results['avg diff err'], 
                       'n_val': 1-abs(self.results['avg diff']), 
                       'n_err': self.results['avg diff err'], 
                       }]
            if self.full_output:
                for model in self.models:
                    mod_dict = self.results[model]
                    output.extend([{'metric': f'{model}_syn_F1', 'dim': 'u',
                        'val': mod_dict['fr_val_acc'], 
                        'err': mod_dict['fr_val_err'], 
                        'n_val': mod_dict['fr_val_acc'], 
                        'n_err': mod_dict['fr_val_err'], 
                        }])
                    if len(self.models) > 1:
                        output.extend([
                            {'metric': f'{model}_F1_diff', 'dim': 'u',
                                'val': mod_dict['fr_val_acc']-mod_dict['rr_val_acc'],
                                'err': np.sqrt(mod_dict['rr_val_err']**2+mod_dict['fr_val_err']**2),
                                'n_val': 1-abs(mod_dict['fr_val_acc']-mod_dict['rr_val_acc']),
                                'n_err': np.sqrt(mod_dict['rr_val_err']**2+mod_dict['fr_val_err']**2),
                            }])
            if (self.hout_data is not None):
                output.extend([{'metric': 'avg_F1_diff_hout', 'dim': 'u',
                       'val': self.results['avg diff hout'], 
                       'err': self.results['avg diff err hout'], 
                       'n_val': 1-abs(self.results['avg diff hout']), 
                       'n_err': self.results['avg diff err hout'], 
                       }])
                if self.full_output:
                    for model in self.models:
                        mod_dict = self.results[model]
                        output.extend([{'metric': f'{model}_syn_F1_hout', 'dim': 'u',
                            'val': mod_dict['fr_test_acc'], 
                            'n_val': mod_dict['fr_test_acc'], 
                            }])
                        if len(self.models) > 1:
                            output.extend([
                                {'metric': f'{model}_F1_diff_hout', 'dim': 'u',
                                    'val': mod_dict['fr_test_acc']-mod_dict['rr_test_acc'],
                                    'n_val': 1-abs(mod_dict['fr_test_acc']-mod_dict['rr_test_acc']),
                                }])
            return output
        else: pass