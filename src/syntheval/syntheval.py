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
from pandas import DataFrame

from .metrics import load_metrics
from .utils.console_output import print_results_to_console
from .utils.preprocessing import consistent_label_encoding
from .utils.postprocessing import extremes_ranking, linear_ranking, quantile_ranking
from .utils.variable_detection import get_cat_variables

loaded_metrics = load_metrics()
#print(loaded_metrics)

def _has_not_slash_backslash_or_dot(input_string):
    return not ('/' in input_string or '\\' in input_string or '.' in input_string)

class SynthEval():
    def __init__(self, 
                 real: DataFrame, 
                 hold_out: DataFrame = None,
                 cat_cols: list = None,
                 nn_distance: str = 'gower', 
                 unique_threshold: int = 10,
                 verbose: bool = True,
        ) -> None:

        self.real = real
        self.verbose = verbose

        if hold_out is not None:
            self.hold_out = hold_out

            # Make sure columns and their order are the same.
            if len(real.columns) == len(hold_out.columns):
                hold_out = hold_out[real.columns.tolist()]
            assert real.columns.tolist() == hold_out.columns.tolist(), 'Columns in real and houldout dataframe are not the same'
        else:
            self.hold_out = None

        if cat_cols is None:
            cat_cols = get_cat_variables(real, unique_threshold)
            if self.verbose: print('SynthEval: inferred categorical columns...')
        
        self.categorical_columns = cat_cols
        self.numerical_columns = [column for column in real.columns if column not in cat_cols]

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

    def evaluate(self, synthetic_dataframe, analysis_target_var=None, presets_file=None, **kwargs):
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

        CLE = consistent_label_encoding(self.real, self.synt, self.categorical_columns, self.hold_out)
        real_data = CLE.encode(self.real)
        synt_data = CLE.encode(self.synt)
        if self.hold_out is not None: hout_data = CLE.encode(self.hold_out)
        else: hout_data = None

        utility_output_txt = ''
        privacy_output_txt = ''

        methods = evaluation_config.keys()

        raw_results = {}
        key_results = None
        scores = {'utility':{'val':[],'err':[]}, 'privacy':{'val':[],'err':[]}}
        pbar = tqdm(methods, disable= not self.verbose)
        for method in pbar:
            pbar.set_description(f'Syntheval: {method}')
            if method not in loaded_metrics.keys():
                print(f"Unrecognised keyword: {method}")
                #raise Exception(f"Unrecognised keyword: {method}")
                continue
            
            M = loaded_metrics[method](real_data, synt_data, hout_data, self.categorical_columns, self.numerical_columns, self.nn_dist, analysis_target_var, do_preprocessing=False, verbose=self.verbose)
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
                    key_results = pd.DataFrame(M.normalize_output(), columns=['metric', 'dim', 'val','err','n_val','n_err','idx_val','idx_err'])
                else:
                    tmp_df = pd.DataFrame(M.normalize_output(), columns=['metric', 'dim', 'val','err','n_val','n_err','idx_val','idx_err'])
                    key_results = pd.concat((key_results,tmp_df), axis = 0).reset_index(drop=True)

        scores = {'utility':{'val': key_results[key_results['dim'] == 'u']['idx_val'].dropna().tolist(),
                             'err': key_results[key_results['dim'] == 'u']['idx_err'].dropna().tolist()},
                  'privacy':{'val': key_results[key_results['dim'] == 'p']['idx_val'].dropna().tolist(),
                             'err': key_results[key_results['dim'] == 'p']['idx_err'].dropna().tolist()}
                             }

        if self.verbose: print_results_to_console(utility_output_txt,privacy_output_txt,scores)

        if (kwargs != {} and self.verbose):
            with open('SE_config.json', "w") as json_file:
                json.dump(evaluation_config, json_file)

        self._raw_results = raw_results
        return key_results

    def benchmark(self, path_to_syn_file_folder, analysis_target_var=None, presets_file=None, rank_strategy='normal', **kwargs):
        """Method for running SynthEval multiple times across all synthetic data files in a
        specified directory. Making a results file, and calculating rank-derived utility 
        and privacy scores.
        
        Takes:
            path_to_syn_file_folder : string like '/example/ex_data_dir/' to folder with datasets
            analysis_target_var     : string column name of categorical variable to check
            rank_strategy           : {default='normal', 'linear', 'quantile'} see descriptions below

        Returns:
            val_df  : dataframe with the metrics and their rank derived scores
            rank_df : dataframe with the ranks used to make the scores
            """

        # Part to avoid printing in the following and resetting to user preference after
        verbose_flag = False
        if self.verbose == True:
            verbose_flag = True
            self.verbose = False

        # Part to loop over all datasets in the folder and evaluate them
        csv_files = sorted(glob.glob(path_to_syn_file_folder + '*.csv'))

        results = {}
        for file in tqdm(csv_files,desc='SynthEval: benchmark'):
            df_syn = pd.read_csv(file)

            results[file.split(os.path.sep)[-1].replace('.csv', '')] = self.evaluate(df_syn,analysis_target_var, presets_file, **kwargs)

        # Part to postprocess the results, format and rank them in a csv file.
        tmp_df = results[list(results.keys())[0]]

        utility_mets = tmp_df[tmp_df['dim'] == 'u']['metric'].tolist()
        privacy_mets = tmp_df[tmp_df['dim'] == 'p']['metric'].tolist()

        vals_df = pd.DataFrame(columns=tmp_df['metric'])
        rank_df = pd.DataFrame(columns=tmp_df['metric'])

        for key, df in results.items():
            df = df.set_index('metric').T

            vals_df.loc[len(vals_df)] = df.loc['val']
            rank_df.loc[len(vals_df)] = df.loc['n_val']

        vals_df['dataset'] = list(results.keys())
        rank_df['dataset'] = list(results.keys())

        vals_df = vals_df.set_index('dataset')
        rank_df = rank_df.set_index('dataset')

        if rank_strategy == 'normal': rank_df = extremes_ranking(rank_df,utility_mets,privacy_mets)
        if rank_strategy == 'linear': rank_df = linear_ranking(rank_df,utility_mets,privacy_mets)
        if rank_strategy == 'quantile': rank_df = quantile_ranking(rank_df,utility_mets,privacy_mets)

        vals_df['rank'] = rank_df['rank']
        vals_df['u_rank'] = rank_df['u_rank']
        vals_df['p_rank'] = rank_df['p_rank']

        name_tag = str(int(time.time()))
        vals_df.to_csv('SE_benchmark_results' +'_' +name_tag+ '.csv')
        vals_df.to_csv('SE_benchmark_ranking' +'_' +name_tag+ '.csv')

        self.verbose = verbose_flag
        return vals_df, rank_df