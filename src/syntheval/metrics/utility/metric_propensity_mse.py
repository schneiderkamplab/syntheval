# Description: propensity mean squared error
# Author: Anton D. Lautrup
# Date: 23-08-2023

import numpy as np
import pandas as pd

from ..core.metric import MetricClass

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import f1_score

from ...utils.preprocessing import stack

class PropensityMeanSquaredError(MetricClass):
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
        return 'p_mse'

    def type() -> str:
        """ Set to 'privacy' or 'utility' """
        return 'utility'

    def evaluate(self,k_folds=5, max_iter=100, solver='liblinear') -> float | dict:
        """Train a a discriminator to distinguish between real and fake data."""

        discriminator = LogisticRegression(max_iter=max_iter, solver=solver, random_state=42)
        Df = stack(self.real_data,self.synt_data).drop(['index'], axis=1)

        Df[self.num_cols] = StandardScaler().fit_transform(Df[self.num_cols])
        Xs, ys = Df.drop(['real'], axis=1), Df['real']

        ### Run 5-fold cross-validation
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        res, acc = [], []
        for train_index, test_index in kf.split(Xs, ys):
            x_train = Xs.iloc[train_index]
            x_test = Xs.iloc[test_index]
            y_train = ys.iloc[train_index]
            y_test = ys.iloc[test_index]

            mod = discriminator.fit(x_train,y_train)
            pred = mod.predict_proba(x_test)
            
            res.append(np.mean((pred[:,0]-0.5)**2))
            acc.append(f1_score(y_test,mod.predict(x_test),average='macro'))

        self.results = {'avg pMSE': np.mean(res), 
                        'pMSE err': np.std(res,ddof=1)/np.sqrt(len(res)),
                        'avg acc': np.mean(acc), 
                        'acc err': np.std(acc,ddof=1)/np.sqrt(len(acc))
                        }
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        string = """\
| Propensity mean squared error (pMSE)     :   %.4f  %.4f   |
|   -> average pMSE classifier accuracy    :   %.4f  %.4f   |""" % (
        self.results['avg pMSE'],
        self.results['pMSE err'],
        self.results['avg acc'],
        self.results['acc err']
        )
        return string

    def normalize_output(self) -> dict:
        """ To add this metric to utility or privacy scores map the main 
        result(s) to the zero one interval where zero is worst performance 
        and one is best.
        
        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err' """

        val_non_lin = np.exp(-10*self.results['avg pMSE'])
        val_non_lin_err = 10*val_non_lin*self.results['pMSE err']

        return {'val': [val_non_lin], 'err': [val_non_lin_err]}
    
    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err idx_val idx_err
            name1  u  0.0  0.0    0.0    0.0    None    None
            name2  p  0.0  0.0    0.0    0.0    0.0     0.0 
        """
        if self.results != {}:
            val_non_lin = np.exp(-10*self.results['avg pMSE'])
            val_non_lin_err = 10*val_non_lin*self.results['pMSE err']

            return [{'metric': 'avg_pMSE', 'dim': 'u', 
                     'val': self.results['avg pMSE'], 
                     'err': self.results['pMSE err'], 
                     'n_val': 1-4*self.results['avg pMSE'], 
                     'n_err': 4*self.results['pMSE err'], 
                     'idx_val': val_non_lin, 
                     'idx_err': val_non_lin_err
                     }]
        else: pass
