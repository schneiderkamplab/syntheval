# Description: Script with for extracting categorical variables automatically.
# Author: Anton D. Lautrup
# Date: 18-08-2023

import numpy as np


def get_cat_variables(df, threshold):
    cat_variables = []

    for col in df.columns:
        if df[col].dtype == "object":
            cat_variables.append(col)
        # https://stackoverflow.com/questions/37726830/how-to-determine-if-a-number-is-any-type-of-int-core-or-numpy-signed-or-not
        elif (
            np.issubdtype(df[col].dtype, np.integer) or np.issubdtype(df[col].dtype, np.floating)
        ) and df[col].nunique() < threshold:
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