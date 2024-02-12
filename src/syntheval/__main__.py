from sys import argv

from click import Path, argument, command, option

from . import SynthEval


@command()
@argument(
    "evaluate",
    required=0
)
@option(
    "--real-data-file",
    "-r",
    type=Path(exists=True),
    help="""Path to csv file with real data.""",
)
@option(
    "--synt-data-file",
    "-s",
    type=Path(exists=True),
    help="""Path to csv file with synthetic data.""",
)
@option(
    "--test-data-file",
    "-h",
    default=None,
    required=0,
    type=Path(exists=True),
    help="""Path to csv file with real data that wasn't used for training.""",
)
@option(
    "--evaluation-config",
    "-j",
    required=0,
    default='full_eval',
    type=Path(exists=True),
    help="""Name of preset file or filepath to custom json config file.""",
)
@option(
    "--category-labels",
    "-l",
    required=0,
    default=None,
    type=Path(exists=True),
    help="""Path to txt file with comma separated labels.""",
)
@option(
    "--class-label",
    "-c",
    default=None,
    type=str,
    help="""Label to use for prediction usability and coloring on plots."""
)
def cli(evaluate,real_data_file,synt_data_file,test_data_file,evaluation_config,category_labels,class_label):
    from syntheval import SynthEval
    ext = real_data_file.split(".")[-1].lower()
    if ext == "csv":
        import pandas as pd
        df_real = pd.read_csv(real_data_file)
        df_fake = pd.read_csv(synt_data_file)
        if test_data_file is not None:
            df_test = pd.read_csv(test_data_file)
        else: df_test = None

    if category_labels is not None:
        if category_labels.split(".")[-1].lower() == "txt":
            with open(category_labels, 'r') as file:
                contents = file.read()
                category_labels = contents.split(',')
        else:
            raise RuntimeError("Please provide category labels in a comma separated txt file")

    evaluator = SynthEval(df_real, holdout_dataframe=df_test, cat_cols=category_labels)
    evaluator.evaluate(df_fake, class_label, evaluation_config)
    
#cli()