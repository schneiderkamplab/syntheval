# Description: Membership Inference Attack based on classification
# Author: Tobias Hyrup
# Date: 2023-10-30

import numpy as np
import pandas as pd

from syntheval.metrics.core.metric import MetricClass

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, f1_score

class AttributeDisclosure(MetricClass):
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
        return "att_discl"

    def type() -> str:
        """Set to 'privacy' or 'utility'"""
        return "privacy"

    def _minmaxscale(self, df: pd.DataFrame, include: list) -> pd.DataFrame:
        """Function for min-max scaling of a dataframe"""
        for column in df.columns:
            if column in include:
                df[column] = (df[column] - df[column].min()) / (
                    df[column].max() - df[column].min()
                )
        return df

    def _predict_cat_target(
        self,
        real: pd.DataFrame,
        syn: pd.DataFrame,
        target: str,
        hout: pd.DataFrame = None,
    ):
        syn_predictors = syn.loc[:, syn.columns != target]
        real_predictors = real.loc[:, real.columns != target]

        syn_target = syn[target]
        real_target = real[target]

        if hout is not None:
            hout_predictors = hout.loc[:, hout.columns != target]
            hout_target = hout[target]

        total_targets = (
            pd.concat([real_target, hout_target]) if hout is not None else real_target
        )
        total_predictors = (
            pd.concat([real_predictors, hout_predictors])
            if hout is not None
            else real_predictors
        )

        clf = RandomForestClassifier(n_estimators=100).fit(syn_predictors, syn_target)

        total_targets = (
            pd.concat([real_target, hout_target]) if hout is not None else real_target
        )
        total_predictors = (
            pd.concat([real_predictors, hout_predictors])
            if hout is not None
            else real_predictors
        )

        preds = clf.predict(total_predictors)
        accuracy = sum(preds == total_targets) / len(total_targets)
        precision = precision_score(
            total_targets, preds, average="macro", zero_division=0
        )
        recall = recall_score(total_targets, preds, average="macro", zero_division=0)
        f1 = f1_score(total_targets, preds, average="macro", zero_division=0)

        return accuracy, precision, recall, f1

    def _predict_num_target(
        self,
        real: pd.DataFrame,
        syn: pd.DataFrame,
        target: str,
        hout: pd.DataFrame = None,
        threshold: float = 1 / 30,
    ):
        syn_target = syn[target]
        real_target = real[target]
        syn_predictors = syn.loc[:, syn.columns != target]
        real_predictors = real.loc[:, real.columns != target]

        if hout is not None:
            hout_predictors = hout.loc[:, hout.columns != target]
            hout_target = hout[target]

        total_targets = (
            pd.concat([real_target, hout_target]) if hout is not None else real_target
        )
        total_predictors = (
            pd.concat([real_predictors, hout_predictors])
            if hout is not None
            else real_predictors
        )

        rf_regressor = RandomForestRegressor(n_estimators=100).fit(
            syn_predictors, syn_target
        )

        preds = rf_regressor.predict(total_predictors)

        # check if differens from preds to target_real is less than the given threshold
        preds = np.where(np.abs(preds - total_targets) <= threshold, 1, 0)
        total_targets = [1] * len(total_targets)

        accuracy = sum(preds == total_targets) / len(total_targets)
        precision = precision_score(
            total_targets, preds, average="macro", zero_division=0
        )
        recall = recall_score(total_targets, preds, average="macro", zero_division=0)
        f1 = f1_score(total_targets, preds, average="macro", zero_division=0)

        return accuracy, precision, recall, f1

    def evaluate(self, numerical_dist_thresh: float = 1 / 30, sensitive: list = None) -> float | dict:
        """Evaluate the attribute disclosure risk of the synthetic data

        Args:
            numerical_dist_thresh (float): Threshold for numerical attributes
            sensitive (list): List of sensitive attributes to evaluate
        
        Returns:
            dict: Dictionary with attribute disclosure risk and standard error
        
        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> fake = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            >>> A = AttributeDisclosure(real, fake, cat_cols=[], num_cols=[], do_preprocessing=False)
            >>> A.evaluate(sensitive=['b']) # doctest: +ELLIPSIS
            {'Attr Dis accuracy': 0.0, ...}
        """
        pre_results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        # Scale the numeric attributes
        if self.hout_data is not None:
            combined_data = pd.concat(
                [self.real_data, self.synt_data, self.hout_data], ignore_index=True
            )
        else:
            combined_data = pd.concat(
                [self.real_data, self.synt_data], ignore_index=True
            )
        scaled = self._minmaxscale(combined_data.copy(deep=True), self.num_cols)

        real_scaled = scaled.iloc[: len(self.real_data)]
        syn_scaled = scaled.iloc[
            len(self.real_data) : len(self.real_data) + len(self.synt_data)
        ]
        hout_scaled = (
            scaled.iloc[len(self.real_data) + len(self.synt_data) :]
            if self.hout_data is not None
            else None
        )

        # Compute attribute disclosure for each attribute with maximum adversarial knowledge
        sensitive_columns = sensitive if sensitive is not None else self.real_data.columns
        for column in sensitive_columns:
            if column in self.cat_cols:
                accuracy, precision, recall, f1 = self._predict_cat_target(
                    real=real_scaled, syn=syn_scaled, hout=hout_scaled, target=column
                )

            else:
                accuracy, precision, recall, f1 = self._predict_num_target(
                    real=real_scaled,
                    syn=syn_scaled,
                    hout=hout_scaled,
                    target=column,
                    threshold=numerical_dist_thresh,
                )

            pre_results["accuracy"].append(accuracy)
            pre_results["precision"].append(precision)
            pre_results["recall"].append(recall)
            pre_results["f1"].append(f1)

        # Compute mean accuracy, precision, recall, and F1-score with accompanying standard errors
        accuracy = np.mean(pre_results["accuracy"])
        accuracy_se = np.std(pre_results["accuracy"], ddof=1) / np.sqrt(
            len(pre_results["accuracy"])
        )
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
            "Attr Dis accuracy": accuracy,
            "Attr Dis accuracy se": accuracy_se,
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
        if self.hout_data is not None:
            string = """\
| Attr. disclosure risk (acc. with holdout):   %.4f  %.4f   |""" % (
            self.results["Attr Dis accuracy"],
            self.results["Attr Dis accuracy se"],
        )
        else:
            string = """\
| Attr. disclosure risk (accuracy)         :   %.4f  %.4f   |""" % (
            self.results["Attr Dis accuracy"],
            self.results["Attr Dis accuracy se"],
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
                    "val": self.results["Attr Dis accuracy"],
                    "err": self.results["Attr Dis accuracy se"],
                    "n_val": 1 - self.results["Attr Dis accuracy"],
                    "n_err": self.results["Attr Dis accuracy se"],
                }
            ]
        else:
            pass
