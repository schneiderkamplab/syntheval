# Description: Script for hosting the main framework
# Author: Anton D. Lautrup
# Date: 16-08-2023

from metrics import *

class SynthEval():
    def __init__(self, unique_threshold = 10, nn_distance = 'gower') -> None:
        pass

    def full_eval(self,real,synt,hout):
        pass

    def fast_eval(self,real,synt,hout):
        pass

    def priv_eval(self,real,synt,hout):
        metrics = ['metric-a', 'metric-b']
        args = [{'arg1':0, 'arg2':1}, {}]
        self.custom_eval(real,synt,hout,metrics,args)
        pass

    def custom_eval(self, real, synt, hout = None, **kwargs):
        
        output_txt = ''

        methods = kwargs.keys()

        # if method not in loaded_metric_classes:
        #     raise Exception("Unrecognised keyword:", method)
        #     continue

        cat_cols = 1
        num_cols = 2

        for metric in loaded_metric_classes:
            if metric.name() not in methods:
                continue
            
            M = metric(real,synt,hout,cat_cols,num_cols)
            _ = M.evaluate(**kwargs[metric.name()])
            #output_txt += M.format_output()

            
        pass


if __name__ == '__main__':
    S = SynthEval()
    S.custom_eval('r','f', None, metric_a={'opt1': 1}, metric_b={})
