# Description: Script for hideing all the ugly console printing stuff from the main SynthEval class 
# Author: Anton D. Lautrup
# Date: 25-08-2023

import numpy as np

def print_results_to_console(utility_output_txt,privacy_output_txt,fairness_output_txt):
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
    if fairness_output_txt != '':
        print("""\
Fairness metric description                   value   error                                 
+---------------------------------------------------------------+"""
            )
        print(fairness_output_txt.rstrip())
        print("""\
+---------------------------------------------------------------+""")

def format_metric_string(name: str, value: float, error: float) -> str:
    """Return string for formatting the output, when the
    metric is part of SynthEval.
    """
    if len(name) >= 40:
        name = name[:39]

    metric_value = f"{value:.4f}"
    metric_error = f"{error:.4f}"

    name = name + ":"
    string = f""
    string += f"| {name:<40}   {metric_value:<7}  {metric_error:<7}   |\n"
    return string
