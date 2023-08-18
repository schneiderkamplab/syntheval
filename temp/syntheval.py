# Description: Script for hosting the main framework
# Author: Anton D. Lautrup
# Date: 16-08-2023

import sys
import json

sys.path.insert(0,'F:/GitHub repositories/syntheval/temp/')
from temp.metrics import load_metrics

from pandas import DataFrame

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

        pass
    
    def full_eval(self,synt):
        with open('temp/presets/full_eval.json','r') as fp:
            loaded_preset = json.load(fp)
        self.custom_eval(synt, **loaded_preset)
        pass

    def fast_eval(self,synt):
        with open('temp/presets/fast_eval.json','r') as fp:
            loaded_preset = json.load(fp)
        self.custom_eval(synt, **loaded_preset)
        pass

    def priv_eval(self,synt):
        with open('temp/presets/privacy.json','r') as fp:
            loaded_preset = json.load(fp)
        self.custom_eval(synt, **loaded_preset)
        pass

    def _update_syn_data(self,synt):
        """Function for adding/updating the synthetic data"""
        self.synt = synt

        if len(self.real.columns) == len(synt.columns):
            synt = synt[self.real.columns.tolist()]
        assert self.real.columns.tolist() == synt.columns.tolist(), 'Columns in real and fake dataframe are not the same'
        print('SynthEval: synthetic data read successfully')
        pass

    def custom_eval(self, synthetic_dataframe, **kwargs):

        self._update_syn_data(synthetic_dataframe)
        
        CLE = consistent_label_encoding(self.real, self.synt, self.categorical_columns, self.hold_out)
        real_data = CLE.encode(self.real)
        synt_data = CLE.encode(self.synt)
        if self.hold_out is not None: hout_data = CLE.encode(self.hold_out)

        output_txt = ''

        methods = kwargs.keys()

        scores = {'utility':[], 'privacy':[]}
        for method in methods:
            if method not in loaded_metrics.keys():
                print(f"Unrecognised keyword: {method}")
                #raise Exception(f"Unrecognised keyword: {method}")
                continue
            
            # if load_metrics[method].type() == 'privacy': privacy_metrics.append(method)
            # if load_metrics[method].type() == 'utility': utility_metrics.append(method)
            
            M = loaded_metrics[method](real_data, synt_data, hout_data, self.categorical_columns, self.numerical_columns, do_preprocessing=False)
            _ = M.evaluate(**kwargs[method])
            output_txt += M.format_output()

            scores[loaded_metrics[method].type()].append(M.normalize_output())

        print(scores)

        print("""\

SynthEval Results
=================================================================

Metric description                            value   error                                 
+---------------------------------------------------------------+"""
        )

        print(output_txt)

        print("""\
+---------------------------------------------------------------+"""
        )
  
        pass


if __name__ == '__main__':
    S = SynthEval('r', cat_cols=1)
    #S.custom_eval('f', metric_a={'opt1': 1}, metric_b={})
    S.priv_eval('f')
