# Description: Membership Inference Attack based on classification
# Author: Tobias Hyrup
# Date: 2023-10-30

import numpy as np
import pandas as pd
from ..core.metric import MetricClass
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder


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
        return "attribute_disclosure_risk"

    def type() -> str:
        """Set to 'privacy' or 'utility'"""
        return "privacy"

    def evaluate(self) -> float | dict:
        """Function for computing the precision, recall, and F1-score of a membership inference attack using a Random Forest classifier"""
        pre_results = {
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for column in self.cat_cols:
            # Remaining categorical columns
            remaining_cat_cols = [c for c in self.cat_cols if c != column]

            combined_data = pd.concat([self.real_data, self.synt_data], ignore_index=True)
            target = combined_data[column]
            predictors = combined_data.loc[:, combined_data.columns != column]

            # Label encode categorical columns
            target = LabelEncoder().fit_transform(target)

            # One-hot encode categorical columns
            combined_data_encoded = pd.get_dummies(
                predictors, columns=remaining_cat_cols, drop_first=True
            )
            
            # Separate into the real and synthetic datasets
            real_enc = combined_data_encoded.iloc[: len(self.real_data)]
            target_real = target[: len(self.real_data)]

            syn_enc = combined_data_encoded.iloc[len(self.real_data) :]
            target_syn = target[len(self.real_data) :]
            
            # Train random forest classifier and predict on real data
            rf_classifier = RandomForestClassifier(n_estimators=100).fit(syn_enc, target_syn)
            preds = rf_classifier.predict(real_enc)

            # Calculate precision, recall, and F1-score
            pre_results["precision"].append(precision_score(target_real, preds, average="macro", zero_division=0))
            pre_results["recall"].append(recall_score(target_real, preds, average="macro", zero_division=0))
            pre_results["f1"].append(f1_score(target_real, preds, average="macro", zero_division=0))

        # Compute mean precision, recall, and F1-score with accompanying standard errors
        precision = np.mean(pre_results["precision"])
        precision_se = np.std(pre_results["precision"]) / np.sqrt(
            len(pre_results["precision"])
        )
        
        recall = np.mean(pre_results["recall"])
        recall_se = np.std(pre_results["recall"]) / np.sqrt(len(pre_results["recall"]))

        f1 = np.mean(pre_results["f1"])
        f1_se = np.std(pre_results["f1"]) / np.sqrt(len(pre_results["f1"]))

        self.results = {
            "Attr Dis precision": precision,
            "Attr Dis precision se": precision_se,
            "Attr Dis recall": recall,
            "Attr Dis recall se": recall_se,
            "Attr Dis macro F1": f1,
            "Attr Dis macro F1 se": f1_se,
        }

        return self.results

    def format_output(self) -> str:
        """Return string for formatting the output, when the
                metric is part of SynthEval.
        |                                          :                    |"""
        string = """\
| Attr Dis macro F1             :   %.4f           |""" % (
            self.results["Attr Dis macro F1"]
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
