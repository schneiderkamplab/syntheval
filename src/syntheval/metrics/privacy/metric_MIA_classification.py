# Description: Membership Inference Attack based on classification
# Author: Tobias Hyrup
# Date: 2023-10-30

import numpy as np
import pandas as pd
from logging import warning
from syntheval.metrics.core.metric import MetricClass
from lightgbm import LGBMClassifier
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
        return "mia"

    def type() -> str:
        """Set to 'privacy' or 'utility'"""
        return "privacy"

    def evaluate(self, num_eval_iter=5) -> float | dict:
        """Function for computing the precision, recall, and F1-score of a membership 
        inference attack using a Random Forest classifier
        
        Args:
            num_eval_iter (int): Number of iterations to run the classifier

        Returns:
            dict: Precision, recall, and F1-score of the membership inference
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> hout = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> M = MIAClassifier(real, fake, hout, cat_cols=[], num_cols=[], do_preprocessing=False)
            >>> M.evaluate(num_eval_iter=1) # doctest: +ELLIPSIS
            {'MIA precision': 0.0, 'MIA precision se': nan, 'MIA recall': 0.0, 'MIA recall se': nan, 'MIA macro F1': 0.33..., 'MIA macro F1 se': nan}
        """
        try:
            assert self.hout_data is not None
        except AssertionError:
            print(" Warning: Membership inference attack metric did not run, holdout data was not supplied!")
            pass
        else:
            if len(self.real_data) < len(self.hout_data) // 2:
                    warning(
                        "The holdout data is more than double the size of the real data. The holdout data will be downsampled to match the size of the real data. real size: %s, holdout size: %s", len(self.real_data), len(self.hout_data)
                    )
            # One-hot encode. All data is combined to ensure consitent encoding
            combined_data = pd.concat(
                [self.real_data, self.synt_data, self.hout_data], ignore_index=True
            )
            combined_data_encoded = pd.get_dummies(
                combined_data, columns=self.cat_cols, drop_first=True
            )

            # Separate into the three datasets
            real = combined_data_encoded.iloc[: len(self.real_data)].reset_index(drop=True)
            syn = combined_data_encoded.iloc[
                len(self.real_data) : len(self.real_data) + len(self.synt_data)
            ].reset_index(drop=True)
            hout = combined_data_encoded.iloc[
                len(self.real_data) + len(self.synt_data) :
            ].reset_index(drop=True)

            # Run classifier multiple times and average the results
            pre_results = {
                "precision": [],
                "recall": [],
                "f1": [],
            }
            for _ in range(num_eval_iter):
                hout_train, hout_test = train_test_split(hout, test_size=0.25)
                syn_samples = syn.sample(n=len(hout_train))

                # Create training data consisting of synthetic and holdout data
                X_train = pd.concat([syn_samples, hout_train], axis=0, ignore_index=True)
                y_train = pd.Series([1] * len(syn_samples) + [0] * len(hout_train))

                # Shuffle
                shuffle_idx = np.arange(len(X_train))
                np.random.shuffle(shuffle_idx)
                X_train = X_train.iloc[shuffle_idx]
                y_train = y_train.iloc[shuffle_idx]
                
                # Create test set by combining some random data from the real and holdout data with an equal number of records from each dataframe
                if len(real) < len(hout_test):
                    # warning(
                    #     "The holdout data is larger than the real data. The holdout data will be downsampled to match the size of the real data."
                    # )
                    hout_sample = hout_test.sample(n=len(real))
                    real_sample = real
                else:
                    real_sample = real.sample(n=len(hout_test))
                    hout_sample = hout_test
                X_test = pd.concat(
                    [
                        real_sample,
                        hout_sample,
                    ],
                    axis=0,
                    ignore_index=True,
                )
                y_test = pd.Series([1] * len(real_sample) + [0] * len(hout_sample))

                cls = LGBMClassifier(verbosity=-1).fit(X_train, y_train)
                # Get predictions
                holdout_predictions = cls.predict(X_test)

                # Calculate precision, recall, and F1-score
                pre_results["precision"].append(
                    precision_score(y_test, holdout_predictions)
                )
                pre_results["recall"].append(recall_score(y_test, holdout_predictions))
                pre_results["f1"].append(
                    f1_score(y_test, holdout_predictions, average="macro")
                )

            precision = np.mean(pre_results["precision"])
            precision_se = np.std(pre_results["precision"], ddof=1) / np.sqrt(
                num_eval_iter
            )

            recall = np.mean(pre_results["recall"])
            recall_se = np.std(pre_results["recall"], ddof=1) / np.sqrt(num_eval_iter)

            f1 = np.mean(pre_results["f1"])
            f1_se = np.std(pre_results["f1"], ddof=1) / np.sqrt(num_eval_iter)

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
        """This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            return [{
                    "metric": "mia_recall",
                    "dim": "p",
                    "val": self.results["MIA recall"],
                    "err": self.results["MIA recall se"],
                    "n_val": 1 - self.results["MIA recall"],
                    "n_err": self.results["MIA recall se"],
                },
                {
                    "metric": "mia_precision",
                    "dim": "p",
                    "val": self.results["MIA precision"],
                    "err": self.results["MIA precision se"],
                    "n_val": 1-2*abs(0.5 - self.results["MIA precision"]),
                    "n_err": self.results["MIA precision se"],
                }]
        else: pass
