# Description: Membership Inference Attack based on classification
# Author: Tobias Hyrup
# Date: 2023-10-30

import numpy as np
import pandas as pd
from ..core.metric import MetricClass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class MIAClassifier(MetricClass):
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
        """Name/keyword to reference the metric"""
        return "mia_risk"

    def type() -> str:
        """Set to 'privacy' or 'utility'"""
        return "privacy"

    def evaluate(self, num_eval_iter=5) -> float | dict:
        """Function for computing the precision, recall, and F1-score of a membership inference attack using a Random Forest classifier"""
        try:
            assert self.hout_data is not None
        except AssertionError:
            print("Error: Membership inference attack metric did not run, holdout data was not supplied!")
            pass
        else:
            # One-hot encode. All data is combined to ensure consitent encoding
            combined_data = pd.concat(
                [self.real_data, self.synt_data, self.hout_data], ignore_index=True
            )
            combined_data_encoded = pd.get_dummies(
                combined_data, columns=self.cat_cols, drop_first=True
            )

            # Separate into the three datasets
            real = combined_data_encoded.iloc[: len(self.real_data)]
            syn = combined_data_encoded.iloc[
                len(self.real_data) : len(self.real_data) + len(self.synt_data)
            ]
            hout = combined_data_encoded.iloc[len(self.real_data) + len(self.synt_data) :]

            # Run classifier multiple times and average the results
            pre_results = {
                "precision": [],
                "recall": [],
                "f1": [],
            }
            for _ in range(num_eval_iter):
                hout_train, hout_test = train_test_split(hout, test_size=0.5)

                # Create training data consisting of synthetic and holdout data
                X_train = pd.concat([syn, hout_train], axis=0)
                y_train = pd.Series([1] * len(syn) + [0] * len(hout_train))

                # Create test set by combining some random data from the real and holdout data with an equal number of records from each dataframe
                X_test = pd.concat(
                    [
                        real.sample(n=len(hout_test)),
                        hout_test,
                    ],
                    axis=0,
                )
                y_test = pd.Series([1] * len(hout_test) + [0] * len(hout_test))

                # Train the classifier on all the data
                rf_classifier = RandomForestClassifier(n_estimators=100).fit(
                    X_train, y_train
                )

                # Get predictions
                holdout_predictions = rf_classifier.predict(X_test)

                # Calculate precision, recall, and F1-score
                pre_results["precision"].append(
                    precision_score(y_test, holdout_predictions)
                )
                pre_results["recall"].append(recall_score(y_test, holdout_predictions))
                pre_results["f1"].append(
                    f1_score(y_test, holdout_predictions, average="macro")
                )

            precision = np.mean(pre_results["precision"])
            precision_se = np.std(pre_results["precision"],ddof=1) / np.sqrt(num_eval_iter)

            recall = np.mean(pre_results["recall"])
            recall_se = np.std(pre_results["recall"],ddof=1) / np.sqrt(num_eval_iter)

            f1 = np.mean(pre_results["f1"])
            f1_se = np.std(pre_results["f1"],ddof=1) / np.sqrt(num_eval_iter)

            self.results = {
                "MIA precision": precision,
                "MIA precision se": precision_se,
                "MIA recall": recall,
                "MIA recall se": recall_se,
                "MIA macro F1": f1,
                "MIA macro F1 se": f1_se,
            }

            return self.results

    def format_output(self) -> str:
        """Return string for formatting the output, when the
                metric is part of SynthEval.
        |                                          :                    |"""
        try:
            assert self.hout_data is not None
        except AssertionError:
            pass
        else:
            string = """\
| Membership inference attack Classifier F1:   %.4f  %.4f   |
|   -> Precision                           :   %.4f  %.4f   |
|   -> Recall                              :   %.4f  %.4f   |""" % (
                self.results["MIA macro F1"],
                self.results["MIA macro F1 se"],
                self.results["MIA precision"],
                self.results["MIA precision se"],
                self.results["MIA recall"],
                self.results["MIA recall se"],
            )
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

            return [{'metric': 'mia_cls_risk', 'dim': 'p', 
                     'val': self.results["MIA recall"], 'err': self.results["MIA recall se"], 
                     'n_val': 1-self.results["MIA recall"], 'n_err': self.results["MIA recall se"]}
                     ]
        else: pass