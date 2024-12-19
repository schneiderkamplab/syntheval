# Description: Script for post-processing results for the benchmark view
# Author: Anton D. Lautrup
# Date: 13-12-2023

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def summation_ranking(dataframe, utility_mets, privacy_mets, fairness_mets):
    """Take the basic sum of the utility and privacy metrics as the rank."""
    res_df = dataframe.copy()

    res_df['rank'] = res_df.sum(axis=1)
    res_df['u_rank'] = res_df[utility_mets].sum(axis=1)
    res_df['p_rank'] = res_df[privacy_mets].sum(axis=1)
    res_df['f_rank'] = res_df[fairness_mets].sum(axis=1)
    return res_df

def linear_ranking(dataframe, utility_mets, privacy_mets, fairness_mets):
    """Apply minmax scaling to the normalised columns, take the row sum as rank.
    Appropriate when there is enough separation between scores that we can trust 
    a linear scale like this."""
    scaler = MinMaxScaler()
    res = scaler.fit_transform(dataframe)
    res_df = pd.DataFrame(res,columns=dataframe.columns,index=dataframe.index)

    res_df['rank'] = res_df.sum(axis=1,numeric_only = True)
    res_df['u_rank'] = res_df[utility_mets].sum(axis=1,numeric_only = True)
    res_df['p_rank'] = res_df[privacy_mets].sum(axis=1,numeric_only = True)
    res_df['f_rank'] = res_df[fairness_mets].sum(axis=1,numeric_only = True)
    return res_df

def extremes_ranking(dataframe, utility_mets, privacy_mets, fairness_mets):
    """Map worst and best score to 0 and 1 respectively, everything else is 0.5.
    This scheme works to separate overall best and worst from normally distributed 
    mass, where we may not be able to say much objectively founded about the
    intermediate results subject to noise."""
    res_df = dataframe.copy()
    res_df[:] = 0.5

    minimum = dataframe.astype(float).min(axis=0)
    maximum = dataframe.astype(float).max(axis=0)
    
    for col in res_df.columns:
        min_con = dataframe[col] == minimum[col]
        max_con = dataframe[col] == maximum[col]

        res_df[col].mask(min_con,0,inplace=True)
        res_df[col].mask(max_con,1,inplace=True)

    res_df['rank'] = res_df.sum(axis=1)
    res_df['u_rank'] = res_df[utility_mets].sum(axis=1)
    res_df['p_rank'] = res_df[privacy_mets].sum(axis=1)
    res_df['f_rank'] = res_df[fairness_mets].sum(axis=1)
    return res_df

def quantile_ranking(dataframe, utility_mets, privacy_mets, fairness_mets):
    """Expermental: Use only if you have enough samples! Sort the results into
    four quantiles and score them 0, 1, 2, 3. Appropriate for lots of samples that
    are not all on top of each other, e.g. high variance, uniform distribution etc."""

    res_df = dataframe.astype(float).apply(lambda x: pd.qcut(x, 4, labels=False, duplicates='drop'))
    
    res_df['rank'] = res_df.sum(axis=1)
    res_df['u_rank'] = res_df[utility_mets].sum(axis=1)
    res_df['p_rank'] = res_df[privacy_mets].sum(axis=1)
    res_df['f_rank'] = res_df[fairness_mets].sum(axis=1)
    return res_df
