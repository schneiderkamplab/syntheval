# Description: Script for hosting the main framework
# Author: Anton D. Lautrup
# Date: 16-08-2023

import json
import glob
import time
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Literal
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
    def __init__(self, 
                 real_dataframe: DataFrame, 
                 holdout_dataframe: DataFrame = None,
                 cat_cols: list = None,
                 nn_distance: Literal['gower', 'euclid', 'EXPERIMENTAL_gower'] = 'gower', 
                 unique_threshold: int = 10,
                 verbose: bool = True,
        ) -> None:
        """Primary object for accessing the SynthEval evaluation framework. Create with the real data used for training 
        and use either evaluate of benchmark methods for evaluating synthetic datasets.
        
        Parameters:
            real_dataframe      : real dataset, in dataframe format. 
            holdout_dataframe   : (optional) real data that was not used for training the generative model
            cat_cols            : (optional) complete list of categorical columns column names. 
            nn_distance         : {default= 'gower', 'euclid', 'EXPERIMENTAL_gower'} distance metric for NN distances.
            unique_threshold    : threshold of unique levels in non-object columns to be considered categoricals.    
            verbose             : flag fo printing to console and making figures
        """

        self.real = real_dataframe
        self.verbose = verbose

        if holdout_dataframe is not None:
            # Make sure columns and their order are the same.
            if len(real_dataframe.columns) == len(holdout_dataframe.columns):
                holdout_dataframe = holdout_dataframe[real_dataframe.columns.tolist()]
            assert real_dataframe.columns.tolist() == holdout_dataframe.columns.tolist(), 'Columns in real and houldout dataframe are not the same'

            self.hold_out = holdout_dataframe
        else:
            self.hold_out = None

        if cat_cols is None:
            cat_cols = get_cat_variables(real_dataframe, unique_threshold)
            if self.verbose: print('SynthEval: inferred categorical columns...')
        
        self.categorical_columns = cat_cols
        self.numerical_columns = [column for column in real_dataframe.columns if column not in cat_cols]

        self.nn_dist = nn_distance
        
        pass

    def _update_syn_data(self, synt):
        """Function for adding/updating the synthetic data"""
        self.synt = synt

        if len(self.real.columns) == len(synt.columns):
            synt = synt[self.real.columns.tolist()]
        assert self.real.columns.tolist() == synt.columns.tolist(), 'Columns in real and fake dataframe are not the same'
        if self.verbose: print('SynthEval: synthetic data read successfully')
        pass

    def display_loaded_metrics(self):
        """Utility function for getting an overview of the loaded modules and their keys."""
        print(loaded_metrics)
        pass

    def evaluate(self, synthetic_dataframe, analysis_target_var=None, presets_file=None, **kwargs):
        """Method for generating the SynthEval evaluation report on a synthetic dataset. Includes the metrics specified in the 
        presets file or through the keyword arguments. Returns a dataframe with the primary results, and prints to console if 
        verbose. The raw output can be accessed as a charateristic of the SynthEval object after running this method 
        (e.i. self._raw_results).
        
        Parameters:
            synthetic_dataframe     : synthetic dataset, in dataframe format. 
            analysis_target_var     : string column name of categorical variable to check.
            presets_file            : {default=None, 'full_eval', 'fast_eval', 'privacy'} or json file path.
            **kwargs                : keyword arguments for metrics e.g. ks_test={}, eps_risk={}, ...

        Returns:
            key_results : dataframe with the primary result(s) from each of the included metrics.
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

        methods = evaluation_config.keys()

        raw_results = {}
        key_results = None
        #scores = {'utility':{'val':[],'err':[]}, 'privacy':{'val':[],'err':[]}}
        pbar = tqdm(methods, disable= not self.verbose)
        for method in pbar:
            pbar.set_description(f'Syntheval: {method}')
            if method not in loaded_metrics.keys():
                print(f"Unrecognised keyword: {method}")
                #raise Exception(f"Unrecognised keyword: {method}")
                continue
            
            M = loaded_metrics[method](real_data, synt_data, hout_data, self.categorical_columns, self.numerical_columns, self.nn_dist, analysis_target_var, do_preprocessing=CLE, verbose=self.verbose)
            raw_results[method] = M.evaluate(**evaluation_config[method])

            string       = M.format_output()
            extra_string = M.extra_formatted_output()
            if string is not None:
                if loaded_metrics[method].type() == 'utility':
                    utility_output_txt += string + '\n'
                    if extra_string is not None: privacy_output_txt += extra_string + '\n'
                else:
                    privacy_output_txt += string + '\n'
                    if extra_string is not None: utility_output_txt += extra_string + '\n'

            normalized_result = M.normalize_output()
            if normalized_result is not None: 
                if key_results is None:
                    key_results = pd.DataFrame(M.normalize_output(), columns=['metric', 'dim', 'val','err','n_val','n_err'])#,'idx_val','idx_err'])
                else:
                    tmp_df = pd.DataFrame(M.normalize_output(), columns=['metric', 'dim', 'val','err','n_val','n_err'])#,'idx_val','idx_err'])
                    key_results = pd.concat((key_results,tmp_df), axis = 0).reset_index(drop=True)

        # scores = {'utility':{'val': key_results[key_results['dim'] == 'u']['idx_val'].dropna().tolist(),
        #                      'err': key_results[key_results['dim'] == 'u']['idx_err'].dropna().tolist()},
        #           'privacy':{'val': key_results[key_results['dim'] == 'p']['idx_val'].dropna().tolist(),
        #                      'err': key_results[key_results['dim'] == 'p']['idx_err'].dropna().tolist()}
        #                      }

        if self.verbose: print_results_to_console(utility_output_txt,privacy_output_txt)#,scores)

        if (kwargs != {} and self.verbose):
            with open('SE_config.json', "w") as json_file:
                json.dump(evaluation_config, json_file)

        self._raw_results = raw_results
        return key_results

    def benchmark(self, dict_or_path_to_syn_file_folder, analysis_target_var=None, presets_file=None, rank_strategy='linear', **kwargs):
        """Method for running SynthEval multiple times across all synthetic data files in a
        specified directory. Making a results file, and calculating rank-derived utility 
        and privacy scores.
        
        Parameters:
            dict_or_path_to_syn_file_folder : dict of dataframes or string like '/example/ex_data_dir/' to folder with datasets
            analysis_target_var             : string column name of categorical variable to check
            rank_strategy                   : {default='linear', 'normal', 'quantile', 'summation'} see descriptions below

        Returns:
            comb_df : dataframe with the metrics and their rank derived scores
            rank_df : dataframe with the ranks used to make the scores
            """

        # Part to avoid printing in the following and resetting to user preference after
        verbose_flag = False
        if self.verbose == True:
            verbose_flag = True
            self.verbose = False

        # Part to make sure input is the correct format
        if isinstance(dict_or_path_to_syn_file_folder, str):
            csv_files = sorted(glob.glob(dict_or_path_to_syn_file_folder + '*.csv'))
            
            df_dict = {}
            for file in csv_files: df_dict[file.split(os.path.sep)[-1].replace('.csv', '')] = pd.read_csv(file)
        elif isinstance(dict_or_path_to_syn_file_folder, dict):
            df_dict = dict_or_path_to_syn_file_folder
        else:
            raise Exception("Error: input was not instance of dictionary or filepath!")

        # Evaluate the datasets in parallel
        from joblib import Parallel, delayed
        res_list = Parallel(n_jobs=-2)(delayed(self.evaluate)(dataframe, analysis_target_var, presets_file, **kwargs) for dataframe in df_dict.values())
        
        results = {}
        for res, key in zip(res_list,list(df_dict.keys())): results[key] = res

        # Part to postprocess the results, format and rank them in a csv file.
        tmp_df = results[list(results.keys())[0]]

        utility_mets = tmp_df[tmp_df['dim'] == 'u']['metric'].tolist()
        privacy_mets = tmp_df[tmp_df['dim'] == 'p']['metric'].tolist()

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

        if rank_strategy == 'normal': rank_df = extremes_ranking(rank_df,utility_mets,privacy_mets)
        if rank_strategy == 'linear': rank_df = linear_ranking(rank_df,utility_mets,privacy_mets)
        if rank_strategy == 'quantile': rank_df = quantile_ranking(rank_df,utility_mets,privacy_mets)
        if rank_strategy == 'summation': rank_df = summation_ranking(rank_df,utility_mets,privacy_mets)

        comb_df = pd.DataFrame()
        for column in vals_df.columns:
            comb_df[(column, 'value')] = vals_df[column]
            comb_df[(column, 'error')] = errs_df[column]
        comb_df.columns = pd.MultiIndex.from_tuples(comb_df.columns)

        comb_df['rank'] = rank_df['rank']
        comb_df['u_rank'] = rank_df['u_rank']
        comb_df['p_rank'] = rank_df['p_rank']

        name_tag = str(int(time.time()))
        temp_df = comb_df.copy()
        temp_df.columns = ['_'.join(col) for col in comb_df.columns.values]
        vals_df.to_csv('SE_benchmark_results' +'_' +name_tag+ '.csv')
        temp_df.to_csv('SE_benchmark_ranking' +'_' +name_tag+ '.csv')

        self.verbose = verbose_flag
        return comb_df, rank_df