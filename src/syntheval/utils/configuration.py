# Description: Script for holding objects relevant for configuring SynthEval 
# Author: Anton D. Lautrup
# Date: 02-04-2026

import json
import warnings

from pandas import DataFrame

class AnalysisConfig:
    """ Advanced class for providing analysis_target configuration beyond just the target variable.

    Args:
        dataset (pd.DataFrame) : the dataset to analyze (used only for inferring variable types)
        target_vars (str|list) : column name(s) of the target variable(s)
        confounder_vars (str|list|dict) : column name(s) of the confounder variable(s)
        sensitive_vars (str|list) : column name(s) of the sensitive variable(s)
        auto_exclusive (bool) : whether to automatically enforce mutual exclusivity of target variables
        save_config_name (str) : save the config to a json file with this name (without .json extension)

    Example:
        SE = SynthEval(train_df, console='rich', enable_plots=False)
        analysis_config = AnalysisConfig(
            dataset=train_df,
            target_vars='Status',
            confounder_vars={'Status': ['Age', 'Tumor_Size']},
            sensitive_vars=['Age']
        )
        SE.evaluate(synt_df, analysis_target = analysis_config)
    """
    def __init__(self, dataset: DataFrame, target_vars: list | str, confounder_vars: list | str | dict = None, 
                 sensitive_vars: list | str = None, auto_exclusive: bool = False):
        self.target_vars = target_vars if isinstance(target_vars, list) else [target_vars]

        self.target_types = {}
        for var in self.target_vars:
            if dataset[var].dtype == "object" or dataset[var].dtype == "int":
                self.target_types[var] = len(dataset[var].unique())
            else:
                self.target_types[var] = "num"

        if isinstance(confounder_vars, str):
            self.confounder_vars = {var: [confounder_vars] for var in self.target_vars}
        elif isinstance(confounder_vars, list):
            self.confounder_vars = {var: list(confounder_vars) for var in self.target_vars}
        elif isinstance(confounder_vars, dict):
            self.confounder_vars = confounder_vars
        else:
            self.confounder_vars = {var: [] for var in self.target_vars}
        
        # remove instances where the key is also in its own confounder list
        for key in self.confounder_vars.keys():
            if key in self.confounder_vars[key]:
                self.confounder_vars[key].remove(key)

        # if auto_exclusive is True, add each target variable to the confounder list of the other target variables to enforce mutual exclusivity.
        if auto_exclusive:
            for var in self.target_vars:
                for other_var in self.target_vars:
                    if var != other_var and var not in self.confounder_vars[other_var]:
                        self.confounder_vars[other_var].append(var)
            
        self.sensitive_vars = sensitive_vars if isinstance(sensitive_vars, list) else [sensitive_vars]
        pass

    def save(self, path: str = "SE_analysis_config"):
        """Function for saving the config to a json file."""
        with open(path + ".json", "w") as json_file:
            json.dump({
                "target_vars": self.target_vars,
                "target_types": self.target_types,
                "confounder_vars": self.confounder_vars,
                "sensitive_vars": self.sensitive_vars
            }, json_file)
    
def _analysis_target_parser(real_data, analysis_target, analysis_target_var = None) -> AnalysisConfig:
    # TODO: remove deprecated argument in future release
    if analysis_target_var is not None:
        warnings.warn(
            "'analysis_target_var' is deprecated and will be removed in a future release. Use 'analysis_target' instead.",
            FutureWarning,
            stacklevel=2,
        )
        if isinstance(analysis_target, str) and analysis_target != analysis_target_var:
            raise ValueError("Received both 'analysis_target' and deprecated 'analysis_target_var' with different values.")
        if analysis_target is None:
            analysis_target = analysis_target_var

    if (isinstance(analysis_target, str)) or (isinstance(analysis_target, list)):
        try:
            assert isinstance(analysis_target, str) and analysis_target.endswith('.json')

            with open(analysis_target, 'r') as json_file:
                config_dict = json.load(json_file)
                analysis_target = AnalysisConfig(
                    dataset=real_data,
                    target_vars=config_dict['target_vars'],
                    confounder_vars=config_dict['confounder_vars'],
                    sensitive_vars=config_dict['sensitive_vars'],
                )
            analysis_target.target_types = config_dict['target_types']

        except (AssertionError, FileNotFoundError):
            analysis_target = AnalysisConfig(dataset=real_data, target_vars=analysis_target, confounder_vars=[], sensitive_vars=[])
            analysis_target.save()
    return analysis_target