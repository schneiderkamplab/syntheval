# Description: File for hosting the functions assosicated with files and dataframes
# Author: Anton D. Lautrup
# Date: 09-03-2023

import csv
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

def stack(real,fake):
    """Function for stacking the real and fake dataframes and adding a column for keeping 
    track of which is which. This is essentially to ease the use of seaborn plots hue."""
    real = pd.concat((real.reset_index(),pd.DataFrame(np.ones(len(real)),columns=['real'])),axis=1)
    fake = pd.concat((fake.reset_index(),pd.DataFrame(np.zeros(len(fake)),columns=['real'])),axis=1)
    return pd.concat((real,fake),ignore_index=True)

# def load(filename: str, load_dir='data/'):
#     """Load csv file to pandas DataFrame. The index column is removed 
#     on load since it is redundant with DataFrame objects."""
#     df = pd.read_csv(load_dir + filename + '.csv')#, index_col=0)
#     df = df.dropna()
#     return df

class consistent_label_encoding():
    def __init__(self, real, fake, categorical_columns, hout=None) -> None:
        joint_dataframe = pd.concat((real.reset_index(),fake.reset_index()),axis=0)
        if hout is not None: joint_dataframe = pd.concat((joint_dataframe.reset_index(),hout.reset_index()),axis=0)

        self.encoder = OrdinalEncoder().fit(joint_dataframe[categorical_columns])
        self.cat_cols = categorical_columns
        pass

    def encode(self, data):
        data = data.copy()
        data[self.cat_cols] = self.encoder.transform(data[self.cat_cols])
        return data

# def convert_nummerical_pair(real,fake,categorical_columns):
#     """Function for turning categorical classes into integers 
#     so we dont get issues with strings."""
#     real,fake = real.copy(),fake.copy()

#     for c in categorical_columns:
#         if real[c].dtype == 'object':
#                 real[c] = pd.factorize(real[c], sort=True)[0]
#                 fake[c] = pd.factorize(fake[c], sort=True)[0]
#     return real, fake

# def convert_nummerical_single(data,categorical_columns):
#     """Function for turning categorical classes into integers 
#     so we dont get issues with strings."""
#     data = data.copy()

#     for c in categorical_columns:
#         if data[c].dtype == 'object':
#                 data[c] = pd.factorize(data[c], sort=True)[0]
#     return data

def empty_dict():
    """Function to initialize the dictionary"""

    dict = {
        'Correlation matrix differences (num only)' : '',
        'Pairwise mutual information difference'    : '',
        'Kolmogorov-Smirnov avg. dist'              : '',
        'Kolmogorov-Smirnov avg. p-val'             : '',
        'Number of significant KS-tests at a=0.05'  : '',
        'Fraction of significant KS-tests at a=0.05': '',
        'Number of significant KS-tests at a=0.05'  : '',
        'Average confidence interval overlap'       : '',
        'Number of non overlapping COIs at 95pct'   : '',
        'Fraction of non-overlapping CIs at 95pct'  : '',
        'Average empirical Hellinger distance'      : '',
        'Propensity Mean Squared Error (pMSE)'      : '',
        'Propensity Mean Squared Error (acc)'       : '',
        'Nearest neighbour adversarial accuracy'    : '',
        'models trained on real data'               : '',
        'models trained on fake data'               : '',
        'f1 difference training data'               : '',
        'model trained on real data on holdout'     : '',
        'model trained on fake data on holdout'     : '',
        'f1 difference holdout data'                : '',
        'Overall utility score'                     : '',
        'Normed distance to closest record (DCR)'   : '',
        'Nearest neighbour distance ratio'          : '',
        'Hitting rate (thres = range(att)/30)'      : '',
        'Privacy loss (NNAA)'                       : '',
        'Privacy loss (NNDR)'                       : '',
        'epsilon identifiability risk'              : ''
    }
    return dict

def create_results_file(results, file_name):
    with open(file_name, 'w', newline='') as f:
        csv.writer(f).writerow(results.keys())
    pass

def add_to_results_file(results, file_name):
    with open(file_name, 'a', newline='') as f:
        csv.writer(f).writerow(results.values())
    pass

# def dat_adapter(df,class_lab_col,old_vars,new_vars):
#     """Function for mapping class labels to integers"""
#     # <ADL 23-11-2022 10:31> Seems we dont need this with 
#     # the _convert_nummerical function in the Table evaluator.
#     df_tmp = df.copy()
#     df_tmp[class_lab_col[0]].replace(old_vars,new_vars, inplace=True)
#     return df_tmp 

# def separate_hout(data,frac):
#     train=data.sample(frac=frac,random_state=42)
#     hout=data.drop(train.index)
#     return train, hout

# def save(filename: str, data, save_dir='data/'):
#     """Save pandas DataFrame to csv file."""
#     data.to_csv(save_dir + filename + '.csv')
#     print('++ dataframe saved ++')
#     pass