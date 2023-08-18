# Description: Script with for extracting categorical variables automatically.
# Author: Anton D. Lautrup
# Date: 18-08-2023

def get_cat_variables(df, threshold):
    cat_variables = []

    for col in df.columns:
        if df[col].dtype == 'object':
            cat_variables.append(col)
        elif (df[col].dtype == 'int64' or df[col].dtype == 'float64') and df[col].nunique() < threshold:
            cat_variables.append(col)

    return cat_variables
