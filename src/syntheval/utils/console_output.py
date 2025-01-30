# Description: Script for hideing all the ugly console printing stuff from the main SynthEval class
# Author: Anton D. Lautrup
# Date: 25-08-2023

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.columns import Columns
from rich.spinner import Spinner
from rich.panel import Panel
from rich.tree import Tree
from rich.layout import Layout
from rich.text import Text
from dataclasses import dataclass

from ..metrics import load_metrics

def print_results_to_console(
    utility_output_txt, privacy_output_txt, fairness_output_txt
):
    print(
        """\

SynthEval results
=================================================================
"""
    )

    if utility_output_txt != "":
        print(
            """\
Utility metric description                    value   error                                 
+---------------------------------------------------------------+"""
        )
        print(utility_output_txt.rstrip())
        print(
            """\
+---------------------------------------------------------------+
    """
        )

    if privacy_output_txt != "":
        print(
            """\
Privacy metric description                    value   error                                 
+---------------------------------------------------------------+"""
        )
        print(privacy_output_txt.rstrip())
        print(
            """\
+---------------------------------------------------------------+
    """
        )
    if fairness_output_txt != "":
        print(
            """\
Fairness metric description                   value   error                                 
+---------------------------------------------------------------+"""
        )
        print(fairness_output_txt.rstrip())
        print(
            """\
+---------------------------------------------------------------+"""
        )


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


@dataclass
class ConsoleOutput:
    _title: str
    _metrics: list

    def __post_init__(self):
        self._console = Console()
        # self.console.show_cursor(True)

        loaded_metrics = load_metrics()
        self._type_mapping = {
            metric: loaded_metrics[metric].type() for metric in self.metrics
        }
        unique_metric_types = sorted(np.unique(list(self._type_mapping.values())).tolist(), reverse=True)

        self._tables = {}
        self._table_rows = {}
        for name in unique_metric_types:
            self._tables[name] = Table(title=f"{name.capitalize()} Metrics")

        for metric_type in self._tables.keys():
            metrics_by_type = [
                metric
                for metric, m_type in self._type_mapping.items()
                if metric_type == m_type
            ]
            self._table_rows[metric_type] = [
                [metric, Spinner(name="dots", style="green"), Spinner(name="dots", style="green"), metric] # [metric, value, error, parent]
                for metric in metrics_by_type
            ]

            self._tables[metric_type] = self.create_table(self._table_rows[metric_type], title=f"{metric_type.capitalize()} Metrics")

        cols = Columns(self.tables.values(), align="center", expand=True, equal=True)
        self._metrics_panel = Panel(cols, title=self.title)
        self._error_messages = ""

        self._output = Layout()
        self._output.split_column(
            Layout(self._metrics_panel, name="metrics"),
            Layout(
                Panel(
                    Columns([Text(self._error_messages), Text("")], expand=False),
                    expand=False,
                    title="Error Messages",
                ),
                name="error messages",
            ),
        )
        self._output["error messages"].visible = False

    @property
    def output(self):
        return self._output

    @property
    def tables(self):
        return self._tables
    
    @property
    def metrics_panel(self):
        return self._metrics_panel

    @property
    def console(self):
        return self._console

    @property
    def metrics(self):
        return self._metrics

    @property
    def title(self):
        return self._title

    @property
    def error_messages(self):
        return self._error_messages

    def create_table(self, rows, title=None):
        table = Table(title=title)
        table.add_column("Metric", justify="left", min_width=20)
        table.add_column("Value", justify="center", min_width=10)
        table.add_column("Error", justify="center", min_width=10)
        for metric, value, error, parent in rows:
            if metric == parent:
                table.add_section()
            table.add_row(metric, value, error)
        return table

    def formart_input(self, value):
        val = f"{value:.4f}" if isinstance(value, (int, float)) else value
        val = val if val is not None else "-"
        val = f"[red]{val}[/red]" if val == "FAILED" else val
        return val

    def insert_values(self, metric, value, error):
        metric_type = self._type_mapping[metric]
        row_idx = [row[0] for row in self._table_rows[metric_type]].index(metric)
        
        value = f"{value:.4f}" if isinstance(value, (int, float)) else value
        value = f"[red]{value}[/red]" if value == "FAILED" else value
        error = f"{error:.4f}" if isinstance(error, (int, float)) else "-"

        self._table_rows[metric_type][row_idx][1] = str(value)
        self._table_rows[metric_type][row_idx][2] = str(error)
        self._tables[metric_type] = self.create_table(self._table_rows[metric_type], title=f"{metric_type.capitalize()} Metrics")
        self._metrics_panel = Panel(Columns(self._tables.values(), align="center", expand=True, equal=True), title=self.title)
        return self._output["metrics"].update(self._metrics_panel)
    

    def insert_row(self, metric, value, error, parent):
        metric_type = self._type_mapping[parent]
        metric_name = f"- {metric}"
        row = [self.formart_input(metric_name), self.formart_input(value), self.formart_input(error), self.formart_input(parent)]

        parent_idx = [row[0] for row in self._table_rows[metric_type]].index(parent)

        self._table_rows[metric_type].insert(parent_idx+1, row)

        self._tables[metric_type] = self.create_table(self._table_rows[metric_type], title=f"{metric_type.capitalize()} Metrics")
        self._metrics_panel = Panel(Columns(self._tables.values(), align="center", expand=True, equal=True), title=self.title)
        return self._output["metrics"].update(self._metrics_panel)

    def old_insert_values(
        self, method: str, value: float, error: float = None, error_message: str = None
    ):
        if method not in self.metrics:
            raise ValueError(f"Method {method} not found in metrics.")
        method_type = self._type_mapping[method]
        table = self.tables[method_type]

        value = f"{value:.4f}" if isinstance(value, (int, float)) else value
        value = f"[red]{value}[/red]" if value == "FAILED" else value

        error = f"{error:.4f}" if isinstance(error, (int, float)) else error

        row_index = table.columns[0]._cells.index(method)
        table.columns[-2]._cells[row_index] = value
        table.columns[-1]._cells[row_index] = error if error is not None else ""
        if error_message is not None:
            self.console.print(error_message)
        return table

    def add_error_message(self, message: str):
        self._output["error messages"].visible = True
        self._error_messages += message + "\n"
        self._output["error messages"].update(
            Panel(
                Columns(
                    [
                        Text(
                            self._error_messages,
                            justify="left",
                            tab_size=4,
                            no_wrap=True,
                        ),
                        Text(""),
                    ]
                ),
                expand=False,
                title="Error Messages",
            )
        )
