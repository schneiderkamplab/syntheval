# Description: Membership Inference Attack based on classification
# Author: Tobias Hyrup
# Date: 2023-10-30

import numpy as np
import pandas as pd
from ..core.metric import MetricClass
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
        return "attr_discl_cats"

    def type() -> str:
        """Set to 'privacy' or 'utility'"""
        return "privacy"

    def minmaxscale(self, df: pd.DataFrame, include: list) -> pd.DataFrame:
        """Function for min-max scaling of a dataframe"""
        for column in df.columns:
            if column in include:
                df[column] = (df[column] - df[column].min()) / (
                    df[column].max() - df[column].min()
                )
        return df

    def predict_cat_target(self, real: pd.DataFrame, syn: pd.DataFrame, target: str):
        syn_predictors = syn.loc[:, syn.columns != target]
        real_predictors = real.loc[:, real.columns != target]

        syn_target = syn[target]
        real_target = real[target]

        combined_target = pd.concat([syn_target, real_target], ignore_index=True)
        combined_target_encoded = LabelEncoder().fit_transform(combined_target)
        syn_target = combined_target_encoded[: len(syn_target)]
        real_target = combined_target_encoded[len(syn_target) :]

        clf = RandomForestClassifier(n_estimators=100).fit(syn_predictors, syn_target)
        preds = clf.predict(real_predictors)
        precision = precision_score(
            real_target, preds, average="macro", zero_division=0
        )
        recall = recall_score(real_target, preds, average="macro", zero_division=0)
        f1 = f1_score(real_target, preds, average="macro", zero_division=0)

        return precision, recall, f1

    def predict_num_target(
        self,
        real: pd.DataFrame,
        syn: pd.DataFrame,
        target: str,
        threshold: float = 1 / 30,
    ):
        syn_target = syn[target]
        real_target = real[target]
        syn_predictors = syn.loc[:, syn.columns != target]
        real_predictors = real.loc[:, real.columns != target]

        rf_regressor = RandomForestRegressor(n_estimators=100).fit(
            syn_predictors, syn_target
        )
        preds = rf_regressor.predict(real_predictors)

        # check if differens from preds to target_real is less than the given threshold
        preds = np.where(np.abs(preds - real_target) < threshold, 1, 0)
        real_target = [1] * len(real_target)

        precision = precision_score(
            real_target, preds, average="macro", zero_division=0
        )
        recall = recall_score(real_target, preds, average="macro", zero_division=0)
        f1 = f1_score(real_target, preds, average="macro", zero_division=0)

        return precision, recall, f1

    def evaluate(self, numerical_dist_thresh: float = 1 / 30) -> float | dict:
        pre_results = {
            "precision": [],
            "recall": [],
            "f1": [],
        }
        
        # Scale the numeric attributes
        combined_data = pd.concat([self.real_data, self.synt_data], ignore_index=True)
        scaled = self.minmaxscale(combined_data.copy(deep=True), self.num_cols)
        real_scaled = scaled.iloc[: len(self.real_data)]
        syn_scaled = scaled.iloc[len(self.real_data) :]

        # Compute attribute disclosure for each attribute with maximum adversarial knowledge
        for column in self.real_data.columns:
            if column in self.cat_cols:
                precision, recall, f1 = self.predict_cat_target(
                    real=real_scaled, syn=syn_scaled, target=column
                )

            else:
                precision, recall, f1 = self.predict_num_target(
                    real=real_scaled,
                    syn=syn_scaled,
                    target=column,
                    threshold=numerical_dist_thresh,
                )

            pre_results["precision"].append(precision)
            pre_results["recall"].append(recall)
            pre_results["f1"].append(f1)

        # Compute mean precision, recall, and F1-score with accompanying standard errors
        precision = np.mean(pre_results["precision"])
        precision_se = np.std(pre_results["precision"], ddof=1) / np.sqrt(
            len(pre_results["precision"])
        )

        recall = np.mean(pre_results["recall"])
        recall_se = np.std(pre_results["recall"], ddof=1) / np.sqrt(
            len(pre_results["recall"])
        )

        f1 = np.mean(pre_results["f1"])
        f1_se = np.std(pre_results["f1"], ddof=1) / np.sqrt(len(pre_results["f1"]))

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
| Attrribute disclosure risk (macro F1)    :   %.4f  %.4f   |
|   -> Precision                           :   %.4f  %.4f   |
|   -> Recall                              :   %.4f  %.4f   |""" % (
            self.results["Attr Dis macro F1"],
            self.results["Attr Dis macro F1 se"],
            self.results["Attr Dis precision"],
            self.results["Attr Dis precision se"],
            self.results["Attr Dis recall"],
            self.results["Attr Dis recall se"],
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
            return [
                {
                    "metric": "att_discl_risk",
                    "dim": "p",
                    "val": self.results["Attr Dis macro F1"],
                    "err": self.results["Attr Dis macro F1 se"],
                    "n_val": 1 - self.results["Attr Dis macro F1"],
                    "n_err": self.results["Attr Dis macro F1 se"],
                }
            ]
        else:
            pass
