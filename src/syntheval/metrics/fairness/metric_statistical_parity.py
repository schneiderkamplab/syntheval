# Description: Membership Inference Attack based on classification
# Author: Tobias Hyrup
# Date: 2023-10-30

import numpy as np
import pandas as pd

from logging import warning
from warnings import warn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from syntheval.metrics.core.metric import MetricClass

from syntheval.utils.console_output import format_metric_string

class StatisticalParity(MetricClass):
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
        return "statistical_parity"

    def type() -> str:
        """Set to 'privacy', 'utility', or 'fairness'"""
        return "fairness"

    @staticmethod
    def statistical_parity(
        X: pd.DataFrame, S: str, preds: np.ndarray | pd.Series, positive_pred: int = 1
    ) -> float:
        """Function for computing the statistical parity difference between the protected and unprotected group. Also known as Demographic Parity.

        Parameters
        ----------
        X : pd.DataFrame
            Predictors
        S : str
            Protected attribute
        preds : np.ndarray | pd.Series
            Predictions from a classifier
        positive_pred : int, optional
            The positive class of the classifier, by default 1

        Returns
        -------
        float
            The statistical parity difference between the protected group and the unprotected group.

        Example:
            >>> import pandas as pd
            >>> import numpy as np
            >>> X = pd.DataFrame({'A': [0, 1, 0, 1], 'B': [1, 0, 1, 0]})
            >>> preds = np.array([0, 1, 0, 1])
            >>> S = 'A'
            >>> StatisticalParity.statistical_parity(X, S, preds)
            1.0
        """
        assert len(X) == len(
            preds
        )  # Check that the length of the data and the predictions are the same
        assert S in X.columns  # Check that the sensitive attribute is in the data
        assert positive_pred in [
            0,
            1,
        ]  # Check that the positive prediction is either 0 or 1

        if len(np.unique(preds)) > 2:
            warn(
                "The predictions are not binary. Running the metric on the positive class."
            )

        difference = preds[X[S] == 1].mean() - preds[X[S] == 0].mean()

        return difference if positive_pred == 1 else -difference

    def evaluate(
        self, protected_attribute: str, positive_class: int, folds: int = 5
    ) -> float | dict:
        """Function for evaluating the statistical parity of a classifier

        Parameters
        ----------
        protected_attribute : str
            Protected attribute to evaluate the fairness of the classifier
        positive_class : int
            The positive class of the classifier
        folds : int, optional
            Number of folds to use in the cross-validation, by default 5

        Returns
        -------
        float | dict
            The statistical parity difference between the protected group and the unprotected group.

        Example:
            >>> import pandas as pd
            >>> real = pd.DataFrame({'A': [0, 1, 0, 1], 'B': [1, 0, 1, 0]})
            >>> fake = pd.DataFrame({'A': [0, 1, 0, 1], 'B': [1, 0, 1, 0]})
            >>> SP = StatisticalParity(real, fake, cat_cols=['A'], analysis_target='B', do_preprocessing=False)
            >>> SP.evaluate('A', 1, folds=2) # doctest: +ELLIPSIS
            {'statistical_parity': -1.0, ...}
        """
        assert (
            protected_attribute in self.synt_data.columns
        ), "The protected attribute is not in the data"
        assert (
            positive_class in self.synt_data[self.analysis_target].unique()
        ), "The positive class is not in the data"
        assert positive_class in [0, 1], "The positive class must be either 0 or 1"

        # Split the data
        X = self.synt_data.loc[:, self.synt_data.columns != self.analysis_target]
        y = self.synt_data[self.analysis_target]

        # Train a classifier for each fold
        statistical_paraty_differences = []
        for train, test in KFold(folds).split(X, y):
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]

            # Train a classifier
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            statistical_paraty_differences.append(
                self.statistical_parity(
                    X_test, protected_attribute, preds, positive_class
                )
            )

        # Compute statistical parity
        self.results["statistical_parity"] = np.mean(statistical_paraty_differences)
        self.results["statistical_parity se"] = np.std(
            statistical_paraty_differences, ddof=1
        ) / np.sqrt(folds)

        return self.results

    def format_output(self) -> str:
        """Return string for formatting the output, when the
                metric is part of SynthEval.
        |                                          :                    |"""
        string = format_metric_string(
            "Statistical Parity difference",
            self.results["statistical_parity"],
            self.results["statistical_parity se"],
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
                    "metric": "statistical_parity",
                    "dim": "f",
                    "val": self.results["statistical_parity"],
                    "err": self.results["statistical_parity se"],
                    "n_val": 1 - abs(self.results["statistical_parity"]),
                    "n_err": self.results["statistical_parity se"],
                }
            ]
        else:
            pass
