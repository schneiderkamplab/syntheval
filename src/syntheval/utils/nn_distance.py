# Description: Script for keeping track of already calculated NNs
# Author: Anton D. Lautrup
# Date: 21-08-2023

# import gower

import numpy as np

from typing import Literal
from sklearn.neighbors import NearestNeighbors

#TODO: Save the NNs, so that they can be reused

def _create_matrix_with_ones(indices, num_rows):
    matrix = np.zeros((len(indices),num_rows), dtype=int)
    for i, index in enumerate(indices):
        matrix[i,index] = 1
    return matrix

### Custom Gower matrix
from sklearn.preprocessing import OrdinalEncoder
from scipy.spatial.distance import cdist

def _gower_matrix_sklearn(data_x, data_y=None, cat_features: list = None, weights=None, num_attribute_ranges=None, nums_metric: Literal['L1', 'EXP_L2'] = 'L1'):
    """Modified version of the python gower distance metric implementation
    url: https://pypi.org/project/gower/"""

    X = data_x
    if data_y is None: Y = data_x 
    else: Y = data_y 

    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

    x_n_rows, x_n_cols = X.shape
    y_n_rows, y_n_cols = Y.shape 
    
    out_shape = np.zeros((x_n_rows, y_n_rows), dtype=np.float32)

    ### Bit to infer, cat_features if nothing is supplied 
    if cat_features is None:
        if not isinstance(X, np.ndarray): 
            is_number = np.vectorize(lambda x: not np.issubdtype(x, np.number))
            cat_features = is_number(X.dtypes)    
        else:
            cat_features = np.zeros(x_n_cols, dtype=bool)
            for col in range(x_n_cols):
                if not np.issubdtype(type(X[0, col]), np.number):
                    cat_features[col]=True
    else:          
        cat_features = np.array(cat_features)

    ### Separate out weights
    if weights is None:
        weights = np.ones(X.shape[1])
            
    weights_cat = weights[cat_features]
    weights_num = weights[np.logical_not(cat_features)]

    ### Subsetting
    Z = np.concatenate((X,Y))
    
    x_index = range(0,x_n_rows)
    y_index = range(x_n_rows,x_n_rows+y_n_rows)
    
    Z_num = Z[:,np.logical_not(cat_features)]
    Z_cat = Z[:,cat_features]

    ### Make the denominator for the nummerical normalisation 
    if num_attribute_ranges is None:
        num_attribute_ranges = np.max(np.stack((np.array(np.ptp(Z_num,axis=0),dtype=np.float64),np.ones(len(weights_num)))),axis=0)

    X_num = Z_num[x_index,]
    Y_num = Z_num[y_index,]

    ### Do the nummerical step
    if not np.array_equal(cat_features,np.ones(X.shape[1])):
        if nums_metric == 'L1':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=1, w=(weights_num/num_attribute_ranges))

        elif nums_metric == 'EXP_L2':
                nums_sum = cdist(X_num.astype(float), Y_num.astype(float), 'minkowski', p=2, w=(weights_num/num_attribute_ranges**2))#/np.sqrt(len(weights_num))

        else: raise NotImplementedError("The keyword literal is not a valid!")
    else: nums_sum = out_shape
    
    ### Do the categorical step
    if not np.array_equal(cat_features,np.zeros(X.shape[1])):
        Z_cat_enc = OrdinalEncoder().fit_transform(Z_cat)

        X_cat = Z_cat_enc[x_index,]
        Y_cat = Z_cat_enc[y_index,]

        cat_sum = cdist(X_cat.astype(int),Y_cat.astype(int), 'hamming', w=weights_cat)*len(weights_cat)
    else: cat_sum = out_shape
    
    return (nums_sum+cat_sum)/weights.sum()



def _knn_distance(a, b, cat_cols, num, metric: Literal['gower', 'euclid', 'EXPERIMENTAL_gower'] = 'gower', weights=None):
    def gower_knn(a, b, bool_cat_cols, gower_variant):
            """Function used for finding nearest neighbours"""
            d = []
            if np.array_equal(a,b):
                matrix = _gower_matrix_sklearn(a, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant)+np.eye(len(a))
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(a))
            else:
                matrix = _gower_matrix_sklearn(a, b, cat_features=bool_cat_cols, weights=weights, nums_metric=gower_variant)
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(b))
            return d

    def eucledian_knn(a, b):
            """Function used for finding nearest neighbours"""
            d = []
            nn = NearestNeighbors(n_neighbors=num+1, metric_params={'w':weights}) #TODO: add num_att_range here as well
            if np.array_equal(a,b):
                nn.fit(a)
                dists, _ = nn.kneighbors(a)
                for i in range(num):
                    d.append(dists[:,1+i])
            else:
                nn.fit(b)
                dists, _ = nn.kneighbors(a)
                for i in range(num):
                    d.append(dists[:,i])
            return d

    if metric=='gower' or metric=='EXPERIMENTAL_gower':
        bool_cat_cols = [col1 in cat_cols for col1 in a.columns]
        num_cols = [col2 for col2 in a.columns if col2 not in cat_cols]
        a[num_cols] = a[num_cols].astype("float")
        b[num_cols] = b[num_cols].astype("float")
        if metric=='gower': return gower_knn(a,b,bool_cat_cols, gower_variant = 'L1')
        else: return gower_knn(a,b,bool_cat_cols, gower_variant='EXP_L2')
    if metric=='euclid':
        return eucledian_knn(a,b)
    else: raise Exception("Unknown metric; options are 'gower' or 'euclid'")
