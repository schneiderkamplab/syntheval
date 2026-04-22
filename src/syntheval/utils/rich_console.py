# Description: Module for console output using Rich library
# Author: T. Hyrup & Anton D. Lautrup
# Date: 02-02-2026

import numpy as np

from typing import List

from rich.console import Console
from rich.table import Table
from rich.columns import Columns
from rich.spinner import Spinner
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from dataclasses import dataclass

def in_notebook():
    """Function to check if we are in a notebook environment.
    credit: https://stackoverflow.com/a/22424821
    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

@dataclass
class RichConsole:
    _metrics: list

    def __post_init__(self):
        self._console = Console()

        self._tables = {}
        self._tables["progress"] = self.runtime_table(self._metrics, title="Progress")

        cols = Columns(self.tables.values(), align="center", expand=True, equal=True)
        self._metrics_panel = Panel(cols, title="SynthEval Results")
        self._error_messages = ""
        self._error_buffer = []
        
        self._table_rows = {}
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
    def error_messages(self):
        return self._error_messages
    
    def fix_the_layout(self):
        """Number of columns in the metrics panel might have changed"""
        cols = Columns(self.tables.values(), align="center", expand=True, equal=True)
        self._metrics_panel = Panel(cols, title="SynthEval Results")
        pass
    
    def runtime_table(self, metrics, title="Progress"):
        table = Table(title=title)
        table.add_column("Keys", justify="left", min_width=10)
        table.add_column("", justify="center", min_width=3)
        for metric in metrics:
            table.add_row(metric, Spinner(name="dots", style="green"))
        return table
        
    def update_runtime_table(self, metric: str, status: str|bool):
        """Update the runtime table spinner to a status message"""
        row_idx = [row for row in self.metrics].index(metric)
        self._tables["progress"].columns[1]._cells[row_idx] = str(status)
        return self._output["metrics"].update(self._metrics_panel)
    
    def hide_runtime_table(self, trigger: bool):
        """Function to remove the progress table from the output"""
        if trigger and "progress" in self._tables:
            del self._tables["progress"]
            self.fix_the_layout()
        return self._output["metrics"].update(self._metrics_panel)

    def update_result_table_rows(self, metric: str, result_rows: List[tuple]):
        """
        Input is a list of tuples (metric, value, error, metric_type)
        we want to check if the metric type tables exists, if not create them
        then we want to add the rows to the correct table
        """
        new_type_flag = False

        # Get unique metric types and check if they exist in self._tables
        unique_metric_types = np.unique([row[0] for row in result_rows])
        for metric_type in unique_metric_types:
            if metric_type not in self._tables:
                new_type_flag = True
                self._table_rows[metric_type] = []
                self._tables[metric_type] = self.create_table(self._table_rows[metric_type], title=f"{metric_type.capitalize()} Metrics")
        
        for row in result_rows:
            self.insert_result(metric, row)

        if new_type_flag:
            self.fix_the_layout()
        pass

    def create_table(self, rows, title=None):
        table = Table(title=title)
        table.add_column("Metric", justify="left", min_width=20)
        table.add_column("Value", justify="center", min_width=8)
        table.add_column("Error", justify="center", min_width=8)
        for metric, value, error, parent in rows:
            if metric == parent:
                table.add_section()
            table.add_row(metric, value, error)
        return table

    def format_input(self, value):
        val = f"{value:.4f}" if np.issubdtype(type(value), np.number) else value
        val = val if val is not None else "-"
        val = f"[red]{val}[/red]" if val == "FAILED" else val
        return val
    
    def insert_result(self, parent: str, row: tuple):
        table_id = row[0]
        metric_text = row[1]

        if row[2] is None:
            metric_text = f"[bold]{metric_text}[/bold]"
            formatted_value = ""
            formatted_error = ""
        else:   
            formatted_value = self.format_input(row[2])
            formatted_error = self.format_input(row[3])

        row_data = [metric_text, formatted_value, formatted_error, parent]

        self._table_rows[table_id].append(row_data)
        self._tables[table_id].add_row(*row_data[:-1])
        return self._output["metrics"].update(self._metrics_panel)
    
    def add_error_message(self, message: str):
        """Buffer error messages; render later after Live ends."""
        self._error_buffer.append(message)
        
    def flush_errors(self, title: str = "Error Messages"):
        """Print buffered errors after Live finishes."""
        if not self._error_buffer:
            return
        body = "\n".join(self._error_buffer)
        self._console.print(Panel(Text(body, justify="left", no_wrap=True), title=title))
    # def add_error_message(self, message: str):
    #     self._output["error messages"].visible = True
    #     self._error_messages += message + "\n"
    #     self._output["error messages"].update(
    #         Panel(
    #             Columns(
    #                 [
    #                     Text(
    #                         self._error_messages,
    #                         justify="left",
    #                         tab_size=4,
    #                         no_wrap=True,
    #                     ),
    #                     Text(""),
    #                 ]
    #             ),
    #             expand=False,
    #             title="Error Messages",
    #         )
    #     )