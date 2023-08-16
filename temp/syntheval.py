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

    def custom_eval(self,real,synt,hout, components: list, **kwargs):
        
        for component in components:
            if component not in loaded_metric_classes:
                continue

            component.evaluate()
        #TODO: get arguments if any inside
        pass
