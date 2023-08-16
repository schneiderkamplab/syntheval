# Description: Script for hosting the main framework
# Author: Anton D. Lautrup
# Date: 16-08-2023

from metrics import *

class SynthEval():
    def __init__(self) -> None:
        pass

    def full_eval():
        pass

    def fast_eval():
        pass

    def priv_eval():
        pass

    def custom_eval(components: list, **kwargs):
        for component in components:
            if component not in loaded_metric_classes:
                continue
            
            component.evaluate()
        #TODO: get arguments if any inside
        pass
