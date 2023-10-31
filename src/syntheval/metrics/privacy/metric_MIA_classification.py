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
        return "mia_classifier"

    def type() -> str:
        """Set to 'privacy' or 'utility'"""
        return "privacy"

    def evaluate(self) -> float | dict:
        """Function for computing the precision, recall, and F1-score of a membership inference attack using a Random Forest classifier"""

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

        # Split the holdout set for training and testing
        hout_train, hout_test = train_test_split(hout, test_size=0.5, random_state=42)
        
        # Create training data consisting of synthetic and holdout data
        X_train = pd.concat([syn, hout_train], axis=0)
        y_train = pd.Series([1] * len(syn) + [0] * len(hout_train))

        # Create test set by combining some random data from the real and holdout data with an equal number of records from each dataframe
        X_test = pd.concat(
            [
                real.sample(n=len(hout_test), random_state=42),
                hout_test,
            ],
            axis=0,
        )
        y_test = pd.Series([1] * len(hout_test) + [0] * len(hout_test))

        # Train the classifier on all the data
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42).fit(
            X_train, y_train
        )

        # Get predictions
        hout_prediction = rf_classifier.predict(X_test)

        # Calculate precision, recall, and F1-score
        precision = precision_score(y_test, hout_prediction)
        recall = recall_score(y_test, hout_prediction)
        f1 = f1_score(y_test, hout_prediction, average="macro")

        self.results = {
            "MIA precision": precision,
            "MIA recall": recall,
            "MIA macro F1": f1,
        }

        return self.results

    def format_output(self) -> str:
        """Return string for formatting the output, when the
                metric is part of SynthEval.
        |                                          :                    |"""
        string = """\
| MIA Classifier F1             :   %.4f           |""" % (
            self.results["MIA macro F1"]
        )
        return string

    def normalize_output(self) -> dict:
        """To add this metric to utility or privacy scores map the main
        result(s) to the zero one interval where zero is worst performance
        and one is best.

        pass or return None if the metric should not be used in such scores.

        Return dictionary of lists 'val' and 'err'"""
        # val_non_lin = np.exp(-5 * self.results["eps_risk"])
        # return {"val": [val_non_lin], "err": [0]}
