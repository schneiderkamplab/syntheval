# Description: Script with the preprocessing steps for metrics to work
# Author: Anton D. Lautrup
# Date: 16-11-2022

import pandas as pd

from sklearn.preprocessing import OrdinalEncoder

class consistent_label_encoding():
    def __init__(self, real, fake, categorical_columns, hout=None) -> None:
        joint_dataframe = pd.concat((real.reset_index(),fake.reset_index()),axis=0)
        if hout is not None: joint_dataframe = pd.concat((joint_dataframe.reset_index(),hout.reset_index()),axis=0)

        self.encoder = OrdinalEncoder().fit(joint_dataframe[categorical_columns])
        self.cat_cols = categorical_columns
        pass

    def encode(self, data):
        data = data.copy()
        data[self.cat_cols] = self.encoder.transform(data[self.cat_cols]).astype('int')
        return data