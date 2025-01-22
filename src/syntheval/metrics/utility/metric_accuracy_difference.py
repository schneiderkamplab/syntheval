# Description: Metric implementation of the classification accuracy difference.
# Author: Anton D. Lautrup
# Date: 22-08-2023

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from syntheval.metrics.core.metric import MetricClass

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
    for r_mod, f_mod in zip(real_models, fake_models):
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

    def evaluate(self, F1_type='micro',k_folds=5) -> dict:
        """ Function for training classifiers
        
        Args:
            F1_type (str): Type of F1 score to compute
            k_folds (int): Number of folds for cross-validation

        Returns:
            dict: Various accuracy and accuracy difference metrics
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3, 2], 'b': [4, 5, 6, 4], 'label': [1, 0, 1, 0]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3, 1], 'b': [4, 5, 6, 1], 'label': [1, 0, 1, 0]})
            >>> cls_acc = ClassificationAccuracy(real, fake, cat_cols=['label'], analysis_target='label', do_preprocessing=False)
            >>> cls_acc.evaluate(k_folds=2) # doctest: +ELLIPSIS
            {'avg diff': 0.0, ...}
        
        """
        try:
            assert self.analysis_target is not None, "Analysis target variable not set!"
            assert self.analysis_target in self.cat_cols, "Analysis target variable not categorical!"
            assert len(self.synt_data[self.analysis_target].unique()) >= 2, "Synthetic label column has less than 2 unique values!"  
        except AssertionError:
            print(" Warning: Classification accuracy metric did not run, analysis target variable not supplied or is not categorical!")
            pass
        else:
            real_x, real_y = self.real_data.drop([self.analysis_target], axis=1), self.real_data[self.analysis_target]
            fake_x, fake_y = self.synt_data.drop([self.analysis_target], axis=1), self.synt_data[self.analysis_target]

            ### Build Classification the models            
            R_DT, F_DT = DecisionTreeClassifier(max_depth=15,random_state=42), DecisionTreeClassifier(max_depth=15,random_state=42)
            R_AB, F_AB = AdaBoostClassifier(n_estimators=10,algorithm='SAMME',learning_rate=1,random_state=42),AdaBoostClassifier(n_estimators=10,algorithm='SAMME',learning_rate=1,random_state=42)
            R_RF, F_RF = RandomForestClassifier(n_estimators=10,max_depth=15, random_state=42),RandomForestClassifier(n_estimators=10,max_depth=15, random_state=42)
            R_LG, F_LG = LogisticRegression(solver='saga', max_iter=5000, random_state=42), LogisticRegression(solver='saga', max_iter=5000, random_state=42)

            real_models = [R_DT, R_AB, R_RF, R_LG]
            fake_models = [F_DT, F_AB, F_RF, F_LG]

            ### Run 5-fold cross-validation
            kf = KFold(n_splits=k_folds)
            res = []
            smol = real_y if len(real_y)<len(fake_y) else fake_y
            for train_index, test_index in tqdm(kf.split(smol), desc='cls_acc', total=k_folds, disable = not self.verbose):
                real_x_train = real_x.iloc[train_index]
                real_x_test = real_x.iloc[test_index]
                real_y_train = real_y.iloc[train_index]
                real_y_test = real_y.iloc[test_index]
                fake_x_train = fake_x.iloc[train_index]
                fake_y_train = fake_y.iloc[train_index]

                res.append(class_test(real_models,fake_models,[real_x_train, real_y_train],
                                                            [fake_x_train, fake_y_train],
                                                            [real_x_test, real_y_test],F1_type))
            
            self.results = {}

            class_avg = np.mean(res,axis=0)
            class_err = np.std(res,axis=0,ddof=1)/np.sqrt(k_folds)
            class_diff = np.abs(class_avg[0,:]-class_avg[1,:])
            class_diff_err = np.sqrt(class_err[0,:]**2+class_err[1,:]**2)

            self.k_folds = k_folds
            self.class_avg = class_avg

            self.results['avg diff'] = np.mean(class_diff)
            self.results['avg diff err'] = 0.25*np.sqrt(sum(class_diff_err**2))
            self.results['diffs'] = class_diff
            self.results['diffs err'] = class_diff_err

            if self.hout_data is not None:
                holdout_res = class_test(real_models, fake_models, [real_x, real_y],
                                                                    [fake_x, fake_y],
                                                                    [self.hout_data.drop([self.analysis_target],axis=1), 
                                                                    self.hout_data[self.analysis_target]],
                                                                    F1_type)
                
                holdout_diff = np.abs(holdout_res[0,:]-holdout_res[1,:])

                self.holdout_res = holdout_res

                self.results['avg diff hout'] = np.mean(holdout_diff)
                self.results['avg diff err hout'] = np.std(holdout_diff,ddof=1)/np.sqrt(4)
                self.results['diffs hout'] = holdout_diff
            return self.results
        
    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval. 
|                                          :                    |"""
        if self.results == {}:
            string = """\
| Classification accuracy [FAILED TO RUN]  :   -.----  -.----   |"""
        else:
            avgs = self.class_avg
            diff = self.results['diffs']
            err = self.results['diffs err']
            string = """\
+---------------------------------------------------------------+

Classification accuracy test     
avg. of %d-fold cross val.:
classifier model             acc_r   acc_f    |diff|  error
+---------------------------------------------------------------+
| DecisionTreeClassifier  :   %.4f  %.4f   %.4f  %.4f   |
| AdaBoostClassifier      :   %.4f  %.4f   %.4f  %.4f   | 
| RandomForestClassifier  :   %.4f  %.4f   %.4f  %.4f   |
| LogisticRegression      :   %.4f  %.4f   %.4f  %.4f   |
+---------------------------------------------------------------+
| Average                 :   %.4f  %.4f   %.4f  %.4f   |\n""" % ( self.k_folds,
avgs[0,0],avgs[1,0],diff[0],err[0], 
avgs[0,1],avgs[1,1],diff[1],err[1],
avgs[0,2],avgs[1,2],diff[2],err[2],
avgs[0,3],avgs[1,3],diff[3],err[3],
np.mean(avgs[0,:]),np.mean(avgs[1,:]),self.results['avg diff'], self.results['avg diff err']
)
        if (self.results != {} and self.hout_data is not None):
            hres = self.holdout_res
            hdiff = self.results['diffs hout']
            string += """\
+---------------------------------------------------------------+

hold out data results:
+---------------------------------------------------------------+
| DecisionTreeClassifier  :   %.4f  %.4f   %.4f           |
| AdaBoostClassifier      :   %.4f  %.4f   %.4f           |
| RandomForestClassifier  :   %.4f  %.4f   %.4f           |
| LogisticRegression      :   %.4f  %.4f   %.4f           |
+---------------------------------------------------------------+
| Average                 :   %.4f  %.4f   %.4f  %.4f   |""" % (
hres[0,0], hres[1,0], hdiff[0],
hres[0,1], hres[1,1], hdiff[1],
hres[0,2], hres[1,2], hdiff[2],
hres[0,3], hres[1,3], hdiff[3],
np.mean(hres[:,0]), np.mean(hres[:,1]), self.results['avg diff hout'], self.results['avg diff err hout']
)
        return string

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).
        
        The required format is:
        metric  val  err  n_val  n_err idx_val idx_err
            name1  0.0  0.0    0.0    0.0    None    None
            name2  0.0  0.0    0.0    0.0    0.0     0.0
        """
        if self.results !={}:
            output = [{'metric': 'cls_F1_diff', 'dim': 'u',
                       'val': self.results['avg diff'], 
                       'err': self.results['avg diff err'], 
                       'n_val': 1-self.results['avg diff'], 
                       'n_err': self.results['avg diff err'], 
                       }]
            if (self.hout_data is not None):
                output.extend([{'metric': 'cls_F1_diff_hout', 'dim': 'u',
                       'val': self.results['avg diff hout'], 
                       'err': self.results['avg diff err hout'], 
                       'n_val': 1-self.results['avg diff hout'], 
                       'n_err': self.results['avg diff err hout'], 
                       }])
            return output
        else: pass