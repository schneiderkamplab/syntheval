# Description: Script for hosting the main framework
# Author: Anton D. Lautrup
# Date: 16-08-2023

import os
import json
import glob
import time

import traceback

import pandas as pd
from tqdm import tqdm
from typing import Literal, List, Dict
from pandas import DataFrame

from .metrics import load_metrics
from .utils.console_output import print_results_to_console
from .utils.preprocessing import consistent_label_encoding
from .utils.postprocessing import extremes_ranking, linear_ranking, quantile_ranking, summation_ranking
from .utils.variable_detection import get_cat_variables

loaded_metrics = load_metrics()
#print(loaded_metrics)

def _has_not_slash_backslash_or_dot(input_string):
    return not ('/' in input_string or '\\' in input_string or '.' in input_string)

class SynthEval():
    """Primary object for accessing the SynthEval evaluation framework. Create with the real data used for training
    and use either evaluate of benchmark methods for evaluating synthetic datasets.

    Attributes:
    real (DataFrame): real dataset, in dataframe format.
    fake (DataFrame): synthetic dataset, in dataframe format.
    holdout (DataFrame): real data that was not used for training the generative model

    categorical_columns (List[str]): complete list of categorical columns column names (inferred if not specified).
    numerical_columns (List[str]): complete list of numerical columns column names (inferred if not specified).

    _raw_results (Dict[str, Any]) : raw output from the metrics.

    """
    def __init__(self, 
                 real_dataframe: DataFrame, 
                 holdout_dataframe: DataFrame = None,
                 cat_cols: List[str] = None,
                 nn_distance: Literal['gower', 'euclid', 'EXPERIMENTAL_gower'] = 'gower', 
                 unique_threshold: int = 10,
                 verbose: bool = True,
        ) -> None:
        """Primary object for accessing the SynthEval evaluation framework. Create with the real data used for training 
        and use either evaluate of benchmark methods for evaluating synthetic datasets.
        
        Args:
            real_dataframe      : real dataset, in dataframe format. 
            holdout_dataframe   : (optional) real data that was not used for training the generative model
            cat_cols            : (optional) complete list of categorical columns column names. 
            nn_distance         : {default= 'gower', 'euclid', 'EXPERIMENTAL_gower'} distance metric for NN distances.
            unique_threshold    : threshold of unique levels in non-object columns to be considered categoricals.    
            verbose             : flag fo printing to console and making figures.
        """

        self.real = real_dataframe
        self.verbose = verbose

        if holdout_dataframe is not None:
            # Make sure columns and their order are the same.
            if len(real_dataframe.columns) == len(holdout_dataframe.columns):
                holdout_dataframe = holdout_dataframe[real_dataframe.columns.tolist()]
            assert real_dataframe.columns.tolist() == holdout_dataframe.columns.tolist(), 'Columns in train and test dataframe are not the same'

            self.hold_out = holdout_dataframe
        else:
            self.hold_out = None

        if cat_cols is None:
            cat_cols = get_cat_variables(real_dataframe, unique_threshold)
            if self.verbose:
                print(f"Inferred categorical columns (unique threshold: {unique_threshold}):\n{cat_cols}")
        
        self.categorical_columns = cat_cols
        self.numerical_columns = [column for column in real_dataframe.columns if column not in cat_cols]

        self.nn_dist = nn_distance
        pass

    def _update_syn_data(self, synt: DataFrame) -> None:
        """Function for adding/updating the synthetic data"""
        self.synt = synt

        if len(self.real.columns) == len(synt.columns):
            synt = synt[self.real.columns.tolist()]
        assert self.real.columns.tolist() == synt.columns.tolist(), 'Columns in real and synthetic dataframe are not the same'
        if self.verbose: print('SynthEval: synthetic data read successfully')
        pass

    def display_loaded_metrics(self):
        """Utility function for getting an overview of the loaded modules and their keys."""
        print(loaded_metrics)
        pass

    def evaluate(self, synthetic_dataframe: DataFrame, analysis_target_var: str = None, presets_file: str = None, **kwargs):
        """Method for generating the SynthEval evaluation report on a synthetic dataset. Includes the metrics specified in the 
        presets file or through the keyword arguments. Returns a dataframe with the primary results, and prints to console if 
        verbose. The raw output can be accessed as a charateristic of the SynthEval object after running this method 
        (e.i. self._raw_results).
        
        Args.:
            synthetic_dataframe     : synthetic dataset, in dataframe format. 
            analysis_target_var     : string column name of categorical variable to check.
            presets_file            : {default=None, 'full_eval', 'fast_eval', 'privacy'} or json file path.
            **kwargs                : keyword arguments for metrics e.g. ks_test={}, eps_risk={}, ...

        Returns:
            key_results : dataframe with the primary result(s) from each of the included metrics.

        Example:
            >>> import pandas as pd
            >>> real_data = pd.read_csv('guides/example/penguins_train.csv')
            >>> synthetic_data = pd.read_csv('guides/example/penguins_BN_syn.csv')
            >>> SE = SynthEval(real_data, verbose = False)
            >>> res = SE.evaluate(synthetic_data, analysis_target_var='species', ks_test={}, eps_risk={})
        """
        self._update_syn_data(synthetic_dataframe)
        
        loaded_preset = {}
        if presets_file is not None:
            ext = presets_file.split(".")[-1].lower()
            if _has_not_slash_backslash_or_dot(presets_file):
                with open(os.path.dirname(__file__)+'/presets/'+presets_file+'.json','r') as fp:
                    loaded_preset = json.load(fp)
            elif (ext == "json"):
                with open(presets_file,'r') as fp:
                    loaded_preset = json.load(fp)
            else:
                raise Exception("Error: unrecognised preset keyword or file format!")
        
        evaluation_config = {**loaded_preset, **kwargs}

        CLE = consistent_label_encoding(self.real, self.synt, self.categorical_columns, self.numerical_columns, self.hold_out)
        real_data = CLE.encode(self.real)
        synt_data = CLE.encode(self.synt)
        if self.hold_out is not None: hout_data = CLE.encode(self.hold_out)
        else: hout_data = None

        utility_output_txt = ''
        privacy_output_txt = ''
        fairness_output_txt = ''

        methods = evaluation_config.keys()

        raw_results = {}
        key_results = None
        pbar = tqdm(methods, disable= not self.verbose)
        for method in pbar:
            pbar.set_description(f'Syntheval: {method}')
            if method not in loaded_metrics.keys():
                print(f"Unrecognised keyword: {method}")
                continue
            try:
                #TODO: Add object manager to increase efficiency by reusing nn distances and trained classification models between metrics. 
                M = loaded_metrics[method](real_data, synt_data, hout_data, self.categorical_columns, self.numerical_columns, self.nn_dist, analysis_target_var, do_preprocessing=CLE, verbose=self.verbose)
                raw_results[method] = M.evaluate(**evaluation_config[method])
            except Exception as e:
                print(f"{method} failed to run. Exception: {e}")
                continue

            string       = M.format_output()
            extra_string = M.extra_formatted_output()

            if string is not None:
                match loaded_metrics[method].type():
                    case 'utility':
                        utility_output_txt += string + '\n'
                    case 'privacy':
                        privacy_output_txt += string + '\n'
                    case 'fairness':
                        fairness_output_txt += string + '\n'

            # TODO: Get rid of the extra_formatted_output thing by using this method instead.
            if extra_string is not None:
                for key in extra_string.keys():
                    match key:
                        case 'utility':
                            utility_output_txt += extra_string[key] + '\n'
                        case 'privacy':
                            privacy_output_txt += extra_string[key] + '\n'
                        case 'fairness':
                            fairness_output_txt += extra_string[key] + '\n'
        
            normalized_result = M.normalize_output()
            if normalized_result is not None: 
                if key_results is None:
                    key_results = pd.DataFrame(M.normalize_output(), columns=['metric', 'dim', 'val','err','n_val','n_err'])
                else:
                    tmp_df = pd.DataFrame(M.normalize_output(), columns=['metric', 'dim', 'val','err','n_val','n_err'])
                    key_results = pd.concat((key_results, tmp_df), axis = 0).reset_index(drop=True)

        if self.verbose: print_results_to_console(utility_output_txt, privacy_output_txt, fairness_output_txt)

        # Save non-standard evaluation config to a json file
        if (kwargs != {} and self.verbose):
            with open('SE_config.json', "w") as json_file:
                json.dump(evaluation_config, json_file)

        self._raw_results = raw_results
        return key_results

    def benchmark(self, dfs_or_path: Dict[str, DataFrame] | str, analysis_target_var=None, presets_file=None, rank_strategy='linear', **kwargs):
        """Method for running SynthEval multiple times across all synthetic data files in a
        specified directory. Making a results file, and calculating rank-derived utility 
        and privacy scores.
        
        Parameters:
            dfs_or_path         : dict of dataframes or string like '/example/ex_data_dir/' to folder with datasets
            analysis_target_var : string column name of categorical variable to check
            rank_strategy       : {default='linear', 'normal', 'quantile', 'summation'} see descriptions below

        Returns:
            comb_df : dataframe with the metrics and their rank derived scores
            rank_df : dataframe with the ranks used to make the scores

        Example:
            >>> import pandas as pd
            >>> real_data = pd.read_csv('guides/example/penguins_train.csv')
            >>> synthetic_data = pd.read_csv('guides/example/penguins_BN_syn.csv')
            >>> SE = SynthEval(real_data, verbose = False)
            >>> res, rank = SE.benchmark({'d1':synthetic_data, 'd2':synthetic_data}, analysis_target_var='species', rank_strategy='summation', p_mse={})
            >>> isinstance(res, pd.DataFrame)
            True
        """

        # Part to avoid printing in the following and resetting to user preference after
        verbose_flag = False
        if self.verbose == True:
            verbose_flag = True
            self.verbose = False

        # Part to process the input
        if isinstance(dfs_or_path, str):
            assert os.path.isdir(dfs_or_path), 'Input is not a valid directory'
            csv_files = sorted(glob.glob(dfs_or_path + '*.csv'))
            
            df_dict = {}
            for file in csv_files: df_dict[file.split(os.path.sep)[-1].replace('.csv', '')] = pd.read_csv(file)
        elif isinstance(dfs_or_path, dict):
            df_dict = dfs_or_path
        else:
            raise Exception("Error: input was not instance of dictionary or filepath!")

        assert len(df_dict) > 0, 'Error: Too few datasets for benchmarking!'

        # Evaluate the datasets in parallel
        from joblib import Parallel, delayed
        res_list = Parallel(n_jobs=-2)(delayed(self.evaluate)(dataframe, analysis_target_var, presets_file, **kwargs) for dataframe in df_dict.values())
        
        results = {}
        for res, key in zip(res_list,list(df_dict.keys())): results[key] = res

        # Part to postprocess the results, format and rank them in a csv file.
        tmp_df = results[list(results.keys())[0]]

        utility_mets = tmp_df[tmp_df['dim'] == 'u']['metric'].tolist()
        privacy_mets = tmp_df[tmp_df['dim'] == 'p']['metric'].tolist()
        fairness_mets = tmp_df[tmp_df['dim'] == 'f']['metric'].tolist()

        vals_df = pd.DataFrame(columns=tmp_df['metric'])
        errs_df = pd.DataFrame(columns=tmp_df['metric'])
        rank_df = pd.DataFrame(columns=tmp_df['metric'])

        for key, df in results.items():
            df = df.set_index('metric').T

            vals_df.loc[len(vals_df)] = df.loc['val']
            errs_df.loc[len(vals_df)] = df.loc['err']
            rank_df.loc[len(vals_df)] = df.loc['n_val']

        vals_df['dataset'] = list(results.keys())
        errs_df['dataset'] = list(results.keys())
        rank_df['dataset'] = list(results.keys())

        vals_df = vals_df.set_index('dataset')
        errs_df = errs_df.set_index('dataset')
        rank_df = rank_df.set_index('dataset')

        match rank_strategy:
            case 'normal': rank_df = extremes_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case'linear': rank_df = linear_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case 'quantile': rank_df = quantile_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case 'summation': rank_df = summation_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case _: raise Exception("Error: unrecognised rank_strategy keyword!")

        comb_df = pd.DataFrame()
        for column in vals_df.columns:
            comb_df[(column, 'value')] = vals_df[column]
            comb_df[(column, 'error')] = errs_df[column]
        comb_df.columns = pd.MultiIndex.from_tuples(comb_df.columns)

        comb_df['rank'] = rank_df['rank']
        comb_df['u_rank'] = rank_df['u_rank']
        comb_df['p_rank'] = rank_df['p_rank']
        comb_df['f_rank'] = rank_df['f_rank']

        name_tag = str(int(time.time()))
        temp_df = comb_df.copy()
        temp_df.columns = ['_'.join(col) for col in comb_df.columns.values]
        vals_df.to_csv('SE_benchmark_results' +'_' +name_tag+ '.csv')
        temp_df.to_csv('SE_benchmark_ranking' +'_' +name_tag+ '.csv')

        self.verbose = verbose_flag
        return comb_df, rank_df