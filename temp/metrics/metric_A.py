# Description: Template script for making new metric classes
# Author: Anton D. Lautrup
# Date: 18-08-2023

from .core.metric import MetricClass

class MetricA(MetricClass):
    def evaluate(self, opt1 = 3):
        print(opt1)
        pass

    def name() -> str:
        return 'metric_a'

    def type() -> str:
        return 'utility'

    def format_output(self) -> str:
        string = """\
|                                          :                    |"""
        return string

    def normalize_output(self):
        return 1






