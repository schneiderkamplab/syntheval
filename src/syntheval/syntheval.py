# Description: Script for hosting the main framework
# Author: Anton D. Lautrup
# Date: 16-08-2023

import sys
import json

import numpy as np

sys.path.insert(0,'F:/GitHub repositories/syntheval/src/syntheval/')
#sys.path.insert(0,'C:/Users/lautrup/Documents/GitHub/syntheval/temp')
from metrics import load_metrics

from pandas import DataFrame

from utils.variable_detection import get_cat_variables
from utils.preprocessing import consistent_label_encoding
from utils.console_output import print_results_to_console

loaded_metrics = load_metrics()
#print(loaded_metrics)

def _has_not_slash_or_backslash(input_string):
    return not ('/' in input_string or '\\' in input_string)

class SynthEval():
    def __init__(self, 
                 real: DataFrame, 
                 hold_out: DataFrame = None,
                 cat_cols: list = None,
                 nn_distance: str = 'gower', 
                 unique_threshold = 10
        ) -> None:

        self.real = real

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
            print('SynthEval: inferred categorical columns...')
        
        self.categorical_columns = cat_cols
        self.numerical_columns = [column for column in real.columns if column not in cat_cols]

        self.nn_dist = nn_distance
        pass
    
    # def full_eval(self, synt, analysis_target_var=None):
    #     with open('temp/presets/full_eval.json','r') as fp:
    #         loaded_preset = json.load(fp)
    #     return self.custom_eval(synt, analysis_target_var, **loaded_preset)

    # def fast_eval(self, synt, analysis_target_var=None):
    #     with open('temp/presets/fast_eval.json','r') as fp:
    #         loaded_preset = json.load(fp)
    #     return self.custom_eval(synt, analysis_target_var, **loaded_preset)

    # def priv_eval(self, synt, analysis_target_var=None):
    #     with open('temp/presets/privacy.json','r') as fp:
    #         loaded_preset = json.load(fp)
    #     return self.custom_eval(synt, analysis_target_var, **loaded_preset)

    def _update_syn_data(self, synt):
        """Function for adding/updating the synthetic data"""
        self.synt = synt

        if len(self.real.columns) == len(synt.columns):
            synt = synt[self.real.columns.tolist()]
        assert self.real.columns.tolist() == synt.columns.tolist(), 'Columns in real and fake dataframe are not the same'
        print('SynthEval: synthetic data read successfully')
        pass

    def evaluate(self, synthetic_dataframe, analysis_target_var=None, presets_file=None, **kwargs):
        self._update_syn_data(synthetic_dataframe)
        
        loaded_preset = {}
        if presets_file is not None:
            if _has_not_slash_or_backslash(presets_file):
                with open('src/syntheval/presets/'+presets_file+'.json','r') as fp:
                    loaded_preset = json.load(fp)
            else:
                with open(presets_file,'r') as fp:
                    loaded_preset = json.load(fp)
        
        evaluation_config = {**loaded_preset, **kwargs}

        CLE = consistent_label_encoding(self.real, self.synt, self.categorical_columns, self.hold_out)
        real_data = CLE.encode(self.real)
        synt_data = CLE.encode(self.synt)
        if self.hold_out is not None: hout_data = CLE.encode(self.hold_out)
        else: hout_data = None

        utility_output_txt = ''
        privacy_output_txt = ''

        methods = evaluation_config.keys()

        results = {}
        scores = {'utility':{'val':[],'err':[]}, 'privacy':{'val':[],'err':[]}}
        for method in methods:
            if method not in loaded_metrics.keys():
                print(f"Unrecognised keyword: {method}")
                #raise Exception(f"Unrecognised keyword: {method}")
                continue
            
            M = loaded_metrics[method](real_data, synt_data, hout_data, self.categorical_columns, self.numerical_columns, self.nn_dist, analysis_target_var, do_preprocessing=False)
            results[method] = M.evaluate(**evaluation_config[method])

            string = M.format_output()
            if string is not None:
                if loaded_metrics[method].type() == 'utility':
                    utility_output_txt += string + '\n'
                else:
                    privacy_output_txt += string + '\n'

            if self.hold_out is not None:
                pl = M.privacy_loss()
                if pl is not None:
                    scores['privacy']["val"].extend(pl[0]["val"])
                    scores['privacy']["err"].extend(pl[0]["err"])
                    privacy_output_txt += pl[1] + '\n'

            normalized_result = M.normalize_output()
            if normalized_result is not None:
                scores[loaded_metrics[method].type()]["val"].extend(normalized_result["val"])
                scores[loaded_metrics[method].type()]["err"].extend(normalized_result["err"])

        print_results_to_console(utility_output_txt,privacy_output_txt,scores)

        if kwargs != {}:
            with open('SE_config.json', "w") as json_file:
                json.dump(evaluation_config, json_file)
        return results
