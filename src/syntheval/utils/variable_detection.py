# Description: Script with for extracting categorical variables automatically.
# Author: Anton D. Lautrup
# Date: 18-08-2023

import pandas as pd


def get_cat_variables(df, threshold):
    cat_variables = []

    for col in df.columns:
        if (
            df[col].dtype == "object"
            or pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
        ):
            cat_variables.append(col)
        # https://stackoverflow.com/questions/37726830/how-to-determine-if-a-number-is-any-type-of-int-core-or-numpy-signed-or-not
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < threshold:
            cat_variables.append(col)

    return cat_variables

def check_missing_values(df, missing_directive):
    match missing_directive:
        case 'raise': 
            try:
                assert not df.isnull().values.any()
            except AssertionError:
                raise ValueError("Missing values found! Please handle missing values before evaluation or set missing_directive to 'ignore' or 'drop'.")
        case 'drop':
            df = df.dropna().reset_index(drop=True)  
        case 'ignore': pass
        case _: raise ValueError("Invalid missing_directive! Please choose from 'raise', 'drop', or 'ignore'.")
    return df
