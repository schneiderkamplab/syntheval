# Description: Script for hosting the main framework
# Author: Anton D. Lautrup
# Date: 16-08-2023

import sys
import json

import numpy as np

sys.path.insert(0,'F:/GitHub repositories/syntheval/temp/')
#sys.path.insert(0,'C:/Users/lautrup/Documents/GitHub/syntheval/temp')
from temp.metrics import load_metrics

from pandas import DataFrame

#from utils.nn_distance import nn_distance_metric
from utils.variable_detection import get_cat_variables
from utils.preprocessing import consistent_label_encoding

loaded_metrics = load_metrics()
print(loaded_metrics)




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
    
    def full_eval(self, synt, analysis_target_var=None):
        with open('temp/presets/full_eval.json','r') as fp:
            loaded_preset = json.load(fp)
        self.custom_eval(synt, analysis_target_var, **loaded_preset)
        pass

    def fast_eval(self,synt, analysis_target_var=None):
        with open('temp/presets/fast_eval.json','r') as fp:
            loaded_preset = json.load(fp)
        self.custom_eval(synt, analysis_target_var, **loaded_preset)
        pass

    def priv_eval(self,synt, analysis_target_var=None):
        with open('temp/presets/privacy.json','r') as fp:
            loaded_preset = json.load(fp)
        self.custom_eval(synt, analysis_target_var, **loaded_preset)
        pass

    def _update_syn_data(self,synt):
        """Function for adding/updating the synthetic data"""
        self.synt = synt

        if len(self.real.columns) == len(synt.columns):
            synt = synt[self.real.columns.tolist()]
        assert self.real.columns.tolist() == synt.columns.tolist(), 'Columns in real and fake dataframe are not the same'
        print('SynthEval: synthetic data read successfully')
        pass

    def format_scores(self, scores):

        print("""\
+---------------------------------------------------------------+"""
        )
        if not scores['utility']['val'] == []:
            scores_lst = np.sqrt(sum(np.square(scores['utility']['err'])))/len(scores['utility']['err'])
            print("""\
| Average utility score (%2d metrics)       :   %.4f  %.4f   |""" % (len(scores['utility']['val']),np.mean(scores['utility']['val']), scores_lst)
            )
        
        if not scores['privacy']['val'] == []:
            scores_lst = np.sqrt(sum(np.square(scores['privacy']['err'])))/len(scores['privacy']['err'])
            print("""\
| Average privacy score (%2d metrics)       :   %.4f  %.4f   |""" % (len(scores['privacy']['val']),np.mean(scores['privacy']['val']), scores_lst)
            )

        print("""\
+---------------------------------------------------------------+"""
        )
        pass

    def custom_eval(self, synthetic_dataframe, analysis_target_var=None, **kwargs):

        self._update_syn_data(synthetic_dataframe)
        
        CLE = consistent_label_encoding(self.real, self.synt, self.categorical_columns, self.hold_out)
        real_data = CLE.encode(self.real)
        synt_data = CLE.encode(self.synt)
        if self.hold_out is not None: hout_data = CLE.encode(self.hold_out)

        utility_output_txt = ''
        privacy_output_txt = ''

        methods = kwargs.keys()

        #nn_obj = nn_distance_metric(real_data, synt_data, self.categorical_columns, self.nn_dist)

        results = {}
        scores = {'utility':{'val':[],'err':[]}, 'privacy':{'val':[],'err':[]}}
        for method in methods:
            if method not in loaded_metrics.keys():
                print(f"Unrecognised keyword: {method}")
                #raise Exception(f"Unrecognised keyword: {method}")
                continue
            
            M = loaded_metrics[method](real_data, synt_data, hout_data, self.categorical_columns, self.numerical_columns, self.nn_dist, analysis_target_var, do_preprocessing=False)
            results[method] = M.evaluate(**kwargs[method])

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

        print("""\

SynthEval results
=================================================================
""")

        if utility_output_txt != '':
            print("""\
Utility metric description                    value   error                                 
+---------------------------------------------------------------+"""
            )
            print(utility_output_txt.rstrip())
            print("""\
+---------------------------------------------------------------+
    """)

        if privacy_output_txt != '':
            print("""\
Privacy metric description                    value   error                                 
+---------------------------------------------------------------+"""
            )
            print(privacy_output_txt.rstrip())
            print("""\
+---------------------------------------------------------------+
    """)

        self.format_scores(scores)

        pass


if __name__ == '__main__':
    S = SynthEval('r', cat_cols=1)
    #S.custom_eval('f', metric_a={'opt1': 1}, metric_b={})
    S.priv_eval('f')