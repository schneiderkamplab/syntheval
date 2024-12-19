# Description: Script with the preprocessing steps for metrics to work
# Author: Anton D. Lautrup
# Date: 16-11-2022

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler


def stack(real, fake):
    """Function for stacking the real and fake dataframes and adding a column for keeping
    track of which is which. This is essentially to ease the use of seaborn plots hue.
    """
    real = pd.concat(
        (real.reset_index(), pd.DataFrame(np.ones(len(real)), columns=["real"])), axis=1
    )
    fake = pd.concat(
        (fake.reset_index(), pd.DataFrame(np.zeros(len(fake)), columns=["real"])),
        axis=1,
    )
    return pd.concat((real, fake), ignore_index=True)


class consistent_label_encoding:
    def __init__(
        self, real, fake, categorical_columns, nummerical_columns, hout=None
    ) -> None:
        assert (
            len(categorical_columns) > 0 or len(nummerical_columns) > 0
        ), "Either categorical or nummerical columns must be provided."

        joint_dataframe = pd.concat((real.reset_index(), fake.reset_index()), axis=0)
        if hout is not None:
            joint_dataframe = pd.concat(
                (joint_dataframe.reset_index(), hout.reset_index()), axis=0
            )

        if len(categorical_columns) > 0:
            self.encoder = OrdinalEncoder().fit(joint_dataframe[categorical_columns])
            self.cat_cols = categorical_columns
        else:
            self.cat_cols = None

        if len(nummerical_columns) > 0:
            self.num_encoder = MinMaxScaler().fit(joint_dataframe[nummerical_columns])
            self.num_cols = nummerical_columns
        else:
            self.num_cols = None
        pass

    def encode(self, data):
        data = data.copy()
        if self.cat_cols is not None:
            data[self.cat_cols] = self.encoder.transform(data[self.cat_cols]).astype(
                "int"
            )
        if self.num_cols is not None:
            data[self.num_cols] = self.num_encoder.transform(data[self.num_cols])
        return data

    def decode(self, data):
        data = data.copy()
        if self.cat_cols is not None:
            data[self.cat_cols] = self.encoder.inverse_transform(data[self.cat_cols])
        if self.num_cols is not None:
            data[self.num_cols] = self.num_encoder.inverse_transform(data[self.num_cols])
        return data