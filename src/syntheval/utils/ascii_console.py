# Description: Script for hideing all the ugly console printing stuff from the main SynthEval class 
# Author: Anton D. Lautrup
# Date: 25-08-2023

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

def format_value(value):
        val = f"{value: .4f}" if isinstance(value, (int, float)) else value
        val = val if val is not None else ""
        return val

def format_metric_string(name: str, value: float, error: float) -> str:
    """Return string for formatting the output, when the
    metric is part of SynthEval.
    """
    if len(name) >= 43:
        name = name[:42] + "."

    metric_value = format_value(value)
    metric_error = format_value(error)

    string = f""
    string += f"| {name:<43}: {metric_value:<7}  {metric_error:<7}  |\n"
    return string

class AsciiConsole:
    def __init__(self):
        self._string_tables = {}
        pass

    def add_results_to_tables(self, result_rows):
        for parent, metric_text, value, error in result_rows:
            if parent not in self._string_tables:
                metric_type = parent + " metric description"
                self._string_tables[parent] = f"{metric_type.capitalize():<42}      value    error\n" + "+" + "-"*64 + "+\n"
            self._string_tables[parent] += format_metric_string(metric_text, value, error)

    def flush_tables(self):
        """Print the tables to the console."""

        print("SynthEval results\n" + "="*66)
        for parent, table in self._string_tables.items():
            print(table + "+" + "-"*64 + "+\n")



