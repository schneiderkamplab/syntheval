# Description: Membership Inference Attack based on classification
# Author: Tobias Hyrup
# Date: 2023-10-30

import numpy as np
import pandas as pd

from warnings import warn

from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from syntheval.metrics.core.metric import MetricClass

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

        Args:
            X (pd.DataFrame): the data to compute the metric on
            S (str): the name of the sensitive attribute column in X
            preds (np.ndarray | pd.Series): the predictions from a classifier to compute the metric on
            positive_pred (int, optional): the positive class of the classifier, either 0 or 1, by default 1
        
        Returns:
            float: the statistical parity difference between the protected and unprotected group
            
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

        # Return a Python float for stable doctest repr across NumPy versions.
        return float(difference if positive_pred == 1 else -difference)

    def evaluate(
        self, positive_class: int = 1, folds: int = 5, full_output: bool = False
    ) -> float | dict:
        """Function for evaluating the statistical parity of a classifier

        Args:
            positive_class (int): The positive class of the classifier Default is 1
            folds (int, optional): Number of folds to use in the cross-validation, by default 5
            full_output (bool, optional): Whether to return full output

        Returns:
            dict: Dictionary with the statistical parity difference and its standard error

        Example:
            >>> import pandas as pd
            >>> from syntheval import AnalysisConfig
            >>> real = pd.DataFrame({'A': [0, 1, 0, 1], 'B': [1, 0, 1, 0], 'label': [0, 1, 0, 1]})
            >>> fake = pd.DataFrame({'A': [0, 1, 0, 1], 'B': [1, 0, 1, 0], 'label': [0, 1, 0, 1]})
            >>> config = AnalysisConfig(dataset=real, target_vars='label', sensitive_vars=['A'])
            >>> SP = StatisticalParity(real, fake, analysis_target=config, do_preprocessing=False)
            >>> SP.evaluate(folds=2) # doctest: +ELLIPSIS
            {'statistical_parity': 1.0, ...
        """
        try:
            assert self.analysis_target is not None, "SynthEval(stat parity): metric did not run, no analysis target variable object specified!"
            assert self.analysis_target.sensitive_vars is not None, "SynthEval(stat parity): metric did not run, no sensitive variable specified!"
            
            target_vars = [
                key for (key, value) in self.analysis_target.target_types.items() 
                if isinstance(value, int) and value == 2
                ]
            
            assert target_vars != [], "SynthEval(stat parity): metric did not run, no categorical target variables with exactly 2 unique values!"
            
            protected_attributes = [var for var in self.analysis_target.sensitive_vars if self.real_data[var].nunique() == 2]
            assert protected_attributes != [], "SynthEval(stat parity): metric did not run, no sensitive variables with exactly 2 unique values!"

            assert positive_class in [0, 1], "SynthEval(stat parity): metric did not run, the positive class argument must be either 0 or 1"
        except AssertionError as e:
            raise ValueError(str(e))

        self.full_output = full_output
        result_rows = []
        for target_var, protected_attribute in product(target_vars, protected_attributes):
            # Drop confounder variables for the current target variable (if any)
            confounders = self.analysis_target.confounder_vars[target_var]
            synt_data = self.synt_data.drop(confounders, axis=1)

            fake_x, fake_y = synt_data.drop([target_var], axis=1), synt_data[target_var]

            # Train a classifier for each fold
            statistical_paraty_differences = []
            for train_idxs, test_idxs in KFold(folds).split(fake_x, fake_y):
                X_train, X_test = fake_x.iloc[train_idxs], fake_x.iloc[test_idxs]
                y_train, y_test = fake_y.iloc[train_idxs], fake_y.iloc[test_idxs]

                # Train a classifier
                clf = RandomForestClassifier(n_estimators=100)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                statistical_paraty_differences.append(
                    self.statistical_parity(
                        X_test, protected_attribute, preds, positive_class
                    )
                )
            target_var = target_var.replace(' ', '_').lower()
            result_rows.append({
                "target_var": target_var,
                "protected_attribute": protected_attribute,
                "statistical_parity": float(np.mean(statistical_paraty_differences)),
                "statistical_parity_se": float(np.std(statistical_paraty_differences, ddof=1) / np.sqrt(folds))
            })

        columns = ["target_var", "protected_attribute", "statistical_parity", "statistical_parity_se"]
        
        self.results["statistical_parity"] = float(np.mean([row["statistical_parity"] for row in result_rows]))
        self.results["statistical_parity_se"] = float(np.sqrt(np.sum([row["statistical_parity_se"]**2 for row in result_rows])) / len(result_rows))
        self.results['raw results'] = pd.DataFrame.from_records(result_rows, columns=columns)
        return self.results

    def format_output(self) -> list:
        """ Return a list of tuples for printing results to the rich console."""
        rows = ('fairness', "Statistical Parity difference", self.results["statistical_parity"], self.results["statistical_parity_se"])
        return [rows]

    def normalize_output(self) -> list:
        """This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            output = [{
                    "metric": "statistical_parity",
                    "dim": "f",
                    "val": self.results["statistical_parity"],
                    "err": self.results["statistical_parity_se"],
                    "n_val": 1 - abs(self.results["statistical_parity"]),
                    "n_err": self.results["statistical_parity_se"],
                }]
            if self.full_output:
                for idx, row in self.results['raw results'].iterrows():
                    output.append({
                        "metric": "sp_" + row["target_var"] + "_" + row["protected_attribute"],
                        "dim": "f",
                        "val": row["statistical_parity"],
                        "err": row["statistical_parity_se"],
                        "n_val": 1 - abs(row["statistical_parity"]),
                        "n_err": row["statistical_parity_se"],
                    })
            return output
        else:
            pass
