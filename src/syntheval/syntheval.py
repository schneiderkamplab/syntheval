# Description: Script for hosting the main framework
# Author: Anton D. Lautrup & T. Hyrup
# Date: 16-08-2023

import os
import json
import glob
import time
import threading

import asyncio
import traceback
from tqdm import tqdm

import pandas as pd
from rich.live import Live
from typing import Literal, List, Dict
from pandas import DataFrame

from .metrics import load_metrics
from .utils.rich_console import RichConsole, in_notebook
from .utils.ascii_console import AsciiConsole
from .utils.preprocessing import consistent_label_encoding
from .utils.configuration import AnalysisConfig, _analysis_target_parser
from .utils.postprocessing import extremes_ranking, linear_ranking, quantile_ranking, summation_ranking
from .utils.variable_detection import get_cat_variables

loaded_metrics = load_metrics()

def _has_not_slash_backslash_or_dot(input_string):
    return not ('/' in input_string or '\\' in input_string or '.' in input_string)

def _metric_work(method, evaluation_config, worker_args):
    try:
        M = method(**worker_args)
        raw_result = M.evaluate(**evaluation_config)
        formatted_output = M.format_output()
        key_result = M.normalize_output()
        error = None
    except Exception as e:
        raw_result, formatted_output, key_result, error = None, None, None, e
    return raw_result, formatted_output, key_result, error

async def _run_metric_with_timeout(method, evaluation_config, worker_args, timeout):
    return await asyncio.wait_for(asyncio.to_thread(_metric_work, method, evaluation_config, worker_args), timeout=timeout)

def _run_coroutine_sync(coro):
    """Run a coroutine from sync code, including notebook environments with a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder = {}
    error_holder = {}

    def _thread_target():
        try:
            result_holder['value'] = asyncio.run(coro)
        except Exception as exc:
            error_holder['error'] = exc

    worker = threading.Thread(target=_thread_target, daemon=True)
    worker.start()
    worker.join()

    if 'error' in error_holder:
        raise error_holder['error']
    return result_holder['value']

def _add_key_results(key_results, key_result):
    if key_result is None:
        return key_results
    if key_results is None:
        key_results = pd.DataFrame(key_result, columns=['metric', 'dim', 'val','err','n_val','n_err'])
    else:
        tmp_df = pd.DataFrame(key_result, columns=['metric', 'dim', 'val','err','n_val','n_err'])
        key_results = pd.concat((key_results, tmp_df), axis = 0).reset_index(drop=True)
    return key_results

class SynthEval():
    """Primary object for accessing the SynthEval evaluation framework. Create with the real data used for training
    and use either evaluate of benchmark methods for evaluating synthetic datasets.

    Attributes:
    real (DataFrame): real dataset, in dataframe format.
    fake (DataFrame): synthetic dataset, in dataframe format.
    holdout (DataFrame): real data that was not used for training the generative model

    categorical_columns (List[str]): complete list of categorical columns column names (inferred if not specified).
    numerical_columns (List[str]): complete list of numerical columns column names (inferred if not specified).

    _raw_results (Dict[str, Any]) : raw output from the metrics.

    """
    def __init__(self, 
                 real_dataframe: DataFrame, 
                 holdout_dataframe: DataFrame = None,
                 cat_cols: List[str] = None,
                 nn_distance: Literal['gower', 'euclid', 'EXPERIMENTAL_gower'] = 'gower', 
                 unique_threshold: int = 10,
                 verbose: bool = True,
                 enable_plots: bool = True,
                 console: Literal['rich', 'ascii', 'off'] = 'rich',
                 timeout: int = None
        ) -> None:
        """Primary object for accessing the SynthEval evaluation framework. Create with the real data used for training 
        and use either evaluate of benchmark methods for evaluating synthetic datasets.
        
        Args:
            real_dataframe      : real dataset, in dataframe format. 
            holdout_dataframe   : (optional) real data that was not used for training the generative model
            cat_cols            : (optional) complete list of categorical columns column names. 
            nn_distance         : {default= 'gower', 'euclid', 'EXPERIMENTAL_gower'} distance metric for NN distances.
            unique_threshold    : threshold of unique levels in non-object columns to be considered categoricals.    
            verbose             : flag for printing heads-up information to the console.
            enable_plots        : flag for enabling plot generation.
            console             : type of console output to use ('rich', 'ascii', 'off').
            timeout             : time in seconds after which a metric evaluation will be interrupted and skipped. Default is None (no timeout).
        """

        self.real = real_dataframe
        self.verbose = verbose
        self.enable_plots = enable_plots

        if in_notebook() and console == 'rich':
            self.console = 'ascii'
            if self.verbose:
                print("Rich console is not supported in this environment. Defaulting to ascii console.")
        else:
            self.console = console
        self.timeout = timeout

        if holdout_dataframe is not None:
            # Make sure columns and their order are the same.
            if len(real_dataframe.columns) == len(holdout_dataframe.columns):
                holdout_dataframe = holdout_dataframe[real_dataframe.columns.tolist()]
            assert real_dataframe.columns.tolist() == holdout_dataframe.columns.tolist(), 'Columns in train and test dataframe are not the same'

            self.hold_out = holdout_dataframe
        else:
            self.hold_out = None
        
        if cat_cols is None:
            cat_cols = get_cat_variables(real_dataframe, unique_threshold)
            if self.verbose:
                print(f"Inferred categorical columns (unique threshold: {unique_threshold}):\n{cat_cols}")
        
        self.categorical_columns = cat_cols
        self.numerical_columns = [column for column in real_dataframe.columns if column not in cat_cols]

        self.nn_dist = nn_distance
        pass

    def _update_syn_data(self, synt: DataFrame) -> None:
        """Function for adding/updating the synthetic data"""
        self.synt = synt

        if len(self.real.columns) == len(synt.columns):
            synt = synt[self.real.columns.tolist()]
        assert self.real.columns.tolist() == synt.columns.tolist(), 'Columns in real and synthetic dataframe are not the same'
        if self.verbose: 
            print('SynthEval: synthetic data read successfully')
        pass

    def display_loaded_metrics(self):
        """Utility function for getting an overview of the loaded modules and their keys."""
        print(loaded_metrics)
        pass

    def evaluate(self, synthetic_dataframe: DataFrame, analysis_target: AnalysisConfig | str = None, presets_file: str = None, **kwargs):
        """Method for generating the SynthEval evaluation report on a synthetic dataset. Includes the metrics specified in the 
        presets file or through the keyword arguments. Returns a dataframe with the primary results, and prints to console if 
        verbose. The raw output can be accessed as a charateristic of the SynthEval object after running this method 
        (e.i. self._raw_results).
        
        Args.:
            synthetic_dataframe     : synthetic dataset, in dataframe format. 
            analysis_target         : string column name of categorical variable to check or an instance of AnalysisConfig.
            presets_file            : {default=None, 'full_eval', 'fast_eval', 'privacy'} or json file path.
            **kwargs                : keyword arguments for metrics e.g. ks_test={}, eps_risk={}, ...

        Deprecated:
            analysis_target_var     : deprecated alias for analysis_target. Will be removed in a future release.
        
        Returns:
            key_results : dataframe with the primary result(s) from each of the included metrics.

        Example:
            >>> import pandas as pd
            >>> real_data = pd.read_csv('guides/example/penguins_train.csv')
            >>> synthetic_data = pd.read_csv('guides/example/penguins_BN_syn.csv')
            >>> SE = SynthEval(real_data, verbose = False, console = 'off', enable_plots = False)
            >>> res = SE.evaluate(synthetic_data, analysis_target = 'species', ks_test={}, eps_risk={})
        """
        self._update_syn_data(synthetic_dataframe)

        # Parse control kwargs before building metric evaluation config.
        analysis_target_var = kwargs.pop('analysis_target_var', None)
        if (analysis_target is not None) or (analysis_target_var is not None):
            analysis_target = _analysis_target_parser(self.real, analysis_target, analysis_target_var)

        metric_kwargs = kwargs
        
        loaded_preset = {}
        if presets_file is not None:
            ext = presets_file.split(".")[-1].lower()
            if _has_not_slash_backslash_or_dot(presets_file):
                with open(os.path.dirname(__file__)+'/presets/'+presets_file+'.json','r') as fp:
                    loaded_preset = json.load(fp)
            elif (ext == "json"):
                with open(presets_file,'r') as fp:
                    loaded_preset = json.load(fp)
            else:
                raise Exception("Error: unrecognised preset keyword or file format!")
        
        evaluation_config = {**loaded_preset, **metric_kwargs}

        CLE = consistent_label_encoding(self.real, self.synt, self.categorical_columns, self.numerical_columns, self.hold_out)
        real_data = CLE.encode(self.real)
        synt_data = CLE.encode(self.synt)
        if self.hold_out is not None: hout_data = CLE.encode(self.hold_out)
        else: hout_data = None

        methods = evaluation_config.keys()

        methods_loaded = []
        for method in methods:
            if method not in loaded_metrics.keys():
                print(f"Unrecognised keyword: {method}")
                continue
            else:
                methods_loaded.append(method)

        worker_args = {
            'real_data': real_data,
            'synt_data': synt_data,
            'hout_data': hout_data,
            'cat_cols': self.categorical_columns,
            'num_cols': self.numerical_columns,
            'nn_dist': self.nn_dist,
            'analysis_target' : analysis_target,
            'do_preprocessing': CLE,
            'verbose': self.verbose,
            'plot_figures': self.enable_plots
        }

        raw_results = {}
        key_results = None

        if self.console == 'rich':
            co = RichConsole(methods_loaded)
            output_screen = co.output
            console = co.console
            error_counter = 0
            with Live(output_screen, console=console, refresh_per_second=4, transient=False, screen=True, vertical_overflow="visible") as live:
                console.show_cursor(True)
                for method in methods_loaded:
                    try:
                        raw, formatted_output, key_result, error = _run_coroutine_sync(
                            _run_metric_with_timeout(
                                loaded_metrics[method], evaluation_config[method], worker_args, self.timeout
                            )
                        )
                        if error is not None:
                            raise error
                        raw_results[method] = raw
                        key_results = _add_key_results(key_results, key_result)

                        if formatted_output is not None:
                            co.update_result_table_rows(method, formatted_output)
                        co.update_runtime_table(method, "[bold green]V[/bold green]")
                    except asyncio.TimeoutError:
                        error_counter += 1
                        co.update_runtime_table(method, "[bold yellow]T[/bold yellow]")
                        continue
                    except Exception as e:
                        error_counter += 1
                        error_message = traceback.format_exc()
                        co.add_error_message(message=error_message)
                        co.update_runtime_table(method, "[bold red]X[/bold red]")
                        continue
                    finally:
                        live.update(output_screen)

            co.hide_runtime_table(trigger = error_counter == 0)
            console.print(output_screen)
            co.flush_errors()
        elif self.console == 'ascii':
            co = AsciiConsole()
            pbar = tqdm(methods_loaded)
            timed_out_methods = [] 
            for method in pbar:
                pbar.set_description(f'Syntheval: {method}')
                try:                    
                    raw, formatted_output, key_result, error = _run_coroutine_sync(
                            _run_metric_with_timeout(
                                loaded_metrics[method], evaluation_config[method], worker_args, self.timeout
                            )
                        )
                    if error is not None:
                        raise error
                    raw_results[method] = raw
                    key_results = _add_key_results(key_results, key_result)
                    
                    if formatted_output is not None:
                            co.add_results_to_tables(formatted_output)
                except asyncio.TimeoutError:
                    timed_out_methods.append(method)
                    continue
                except Exception as e:
                    print(f"{method} failed to run. excpetion: {e}")
                    continue

            co.flush_tables()
            if timed_out_methods != []:
                print(f"Some methods timed out after {self.timeout} seconds:\n{', '.join(timed_out_methods)}")
        else:
            timed_out_methods = [] 
            for method in methods_loaded:
                try:
                    raw, formatted_output, key_result, error = _run_coroutine_sync(
                            _run_metric_with_timeout(
                                loaded_metrics[method], evaluation_config[method], worker_args, self.timeout
                            )
                        )
                    if error is not None:
                        raise error
                    raw_results[method] = raw
                    key_results = _add_key_results(key_results, key_result)
                except asyncio.TimeoutError:
                    timed_out_methods.append(method)
                    continue
                except Exception as e:
                    print(f"{method} failed to run. excpetion: {e}")
                    continue
            if timed_out_methods != []:
                print(f"Some methods timed out after {self.timeout} seconds:\n{', '.join(timed_out_methods)}")

        # Save non-standard evaluation config to a json file
        if (metric_kwargs != {} and self.verbose):
            with open('SE_metrics_config.json', "w") as json_file:
                json.dump(evaluation_config, json_file)

        self.analysis_target_config = analysis_target
        self._raw_results = raw_results
        return key_results

    def benchmark(self, dfs_or_path: Dict[str, DataFrame] | str, analysis_target=None, presets_file=None, rank_strategy='summation', **kwargs):
        """Method for running SynthEval multiple times across all synthetic data files in a
        specified directory. Making a results file, and calculating rank-derived utility 
        and privacy scores.
        
        Parameters:
            dfs_or_path         : dict of dataframes or string like '/example/ex_data_dir/' to folder with datasets
            analysis_target     : string column name of categorical variable to check, or an AnalysisConfig instance
            rank_strategy       : {default='summation', 'normal', 'quantile', 'linear'}, see descriptions below.

        Deprecated:
            analysis_target_var : deprecated alias for analysis_target. Will be removed in a future release.

        Details on rank strategies:
            "summation": Uses default normalisation sums the normalised numbers. A higher number is better.
            "linear": Apply min-max scaling to the normalised columns, take the row sum as rank. Appropriate when there is enough separation between scores that we can trust a linear scale like this.
            "quantile": The normalised numbers are converted to quantiles and then summed. Appropriate for lots of samples that are not all on top of each other, e.g. high variance, uniform distribution etc.
            "normal": Map worst and best score to 0 and 1 respectively, everything else is 0.5. This scheme works to separate overall best and worst from normally distributed mass, where we may not be able to say much objectively founded about the intermediate results subject to noise.
            
        Returns:
            comb_df : dataframe with the metrics and their rank derived scores
            rank_df : dataframe with the ranks used to make the scores

        Example:
            >>> import pandas as pd
            >>> real_data = pd.read_csv('guides/example/penguins_train.csv')
            >>> synthetic_data = pd.read_csv('guides/example/penguins_BN_syn.csv')
            >>> SE = SynthEval(real_data, verbose = False)
            >>> res, rank = SE.benchmark({'d1':synthetic_data, 'd2':synthetic_data}, analysis_target='species', rank_strategy='summation', p_mse={})
            >>> isinstance(res, pd.DataFrame)
            True
        """
        # TODO: integrate the rich console with this method as well.
        # Part to avoid printing in the following and resetting to user preference after
        reset_verbose, reset_plotting, reset_console = self.verbose, self.enable_plots, self.console
        self.verbose, self.enable_plots, self.console = False, False, 'off'

        # Part to process the input
        if isinstance(dfs_or_path, str):
            assert os.path.isdir(dfs_or_path), 'Input is not a valid directory'
            csv_files = sorted(glob.glob(dfs_or_path + '*.csv'))
            
            df_dict = {}
            for file in csv_files: df_dict[file.split(os.path.sep)[-1].replace('.csv', '')] = pd.read_csv(file)
        elif isinstance(dfs_or_path, dict):
            df_dict = dfs_or_path
        else:
            raise Exception("Error: input was not instance of dictionary or filepath!")

        assert len(df_dict) > 0, 'Error: Too few datasets for benchmarking!'

        # Parser for argument type flexibility and deprecation handling for the analysis target variable(s).
        analysis_target_var = kwargs.pop('analysis_target_var', None)
        if (analysis_target is not None) or (analysis_target_var is not None):
            analysis_target = _analysis_target_parser(self.real, analysis_target, analysis_target_var)

        metric_kwargs = kwargs

        # Evaluate the datasets in parallel
        from joblib import Parallel, delayed
        res_list = Parallel(n_jobs=-2)(
            delayed(self.evaluate)(dataframe, analysis_target, presets_file, **metric_kwargs)
            for dataframe in df_dict.values()
        )
        
        results = {}
        for res, key in zip(res_list,list(df_dict.keys())): results[key] = res

        # Part to postprocess the results, format and rank them in a csv file.
        tmp_df = results[list(results.keys())[0]]

        utility_mets = tmp_df[tmp_df['dim'] == 'u']['metric'].tolist()
        privacy_mets = tmp_df[tmp_df['dim'] == 'p']['metric'].tolist()
        fairness_mets = tmp_df[tmp_df['dim'] == 'f']['metric'].tolist()

        vals_df = pd.DataFrame(columns=tmp_df['metric'])
        errs_df = pd.DataFrame(columns=tmp_df['metric'])
        rank_df = pd.DataFrame(columns=tmp_df['metric'])

        for key, df in results.items():
            df = df.set_index('metric').T

            vals_df.loc[len(vals_df)] = df.loc['val']
            errs_df.loc[len(vals_df)] = df.loc['err']
            rank_df.loc[len(vals_df)] = df.loc['n_val']

        vals_df['dataset'] = list(results.keys())
        errs_df['dataset'] = list(results.keys())
        rank_df['dataset'] = list(results.keys())

        vals_df = vals_df.set_index('dataset')
        errs_df = errs_df.set_index('dataset')
        rank_df = rank_df.set_index('dataset')

        match rank_strategy:
            case 'normal': rank_df = extremes_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case 'linear': rank_df = linear_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case 'quantile': rank_df = quantile_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case 'summation': rank_df = summation_ranking(rank_df, utility_mets, privacy_mets, fairness_mets)
            case _: raise Exception("Error: unrecognised rank_strategy keyword!")

        comb_df = pd.DataFrame()
        for column in vals_df.columns:
            comb_df[(column, 'value')] = vals_df[column]
            comb_df[(column, 'error')] = errs_df[column]
        comb_df.columns = pd.MultiIndex.from_tuples(comb_df.columns)

        comb_df['rank'] = rank_df['rank']
        if utility_mets != []:
            comb_df['u_rank'] = rank_df['u_rank']
        if privacy_mets != []:
            comb_df['p_rank'] = rank_df['p_rank']
        if fairness_mets != []:
            comb_df['f_rank'] = rank_df['f_rank']

        name_tag = str(int(time.time()))
        temp_df = comb_df.copy()
        temp_df.columns = ['_'.join(col) for col in comb_df.columns.values]
        vals_df.to_csv('SE_benchmark_results' +'_' +name_tag+ '.csv')
        temp_df.to_csv('SE_benchmark_ranking' +'_' +name_tag+ '.csv')

        self.verbose, self.enable_plots, self.console = reset_verbose, reset_plotting, reset_console
        return comb_df, rank_df