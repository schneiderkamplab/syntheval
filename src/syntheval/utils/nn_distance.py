# Description: Script for keeping track of already calculated NNs
# Author: Anton D. Lautrup
# Date: 21-08-2023

import gower

import numpy as np

from sklearn.neighbors import NearestNeighbors

def _create_matrix_with_ones(indices, num_rows):
    matrix = np.zeros((len(indices),num_rows), dtype=int)
    for i, index in enumerate(indices):
        matrix[i,index] = 1
    return matrix

def _knn_distance(a, b, cat_cols, num, metric='gower', weights=None):
    def gower_knn(a, b, bool_cat_cols):
            """Function used for finding nearest neighbours"""
            d = []
            if np.array_equal(a,b):
                matrix = gower.gower_matrix(a,cat_features=bool_cat_cols,weight=weights)+np.eye(len(a))
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(a))
            else:
                matrix = gower.gower_matrix(a,b,cat_features=bool_cat_cols,weight=weights)
                for _ in range(num):
                    d.append(matrix.min(axis=1))
                    matrix += _create_matrix_with_ones(matrix.argmin(axis=1,keepdims=True),len(b))
            return d

    def eucledian_knn(a, b):
            """Function used for finding nearest neighbours"""
            d = []
            nn = NearestNeighbors(n_neighbors=num+1, metric_params={'w':weights})
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

    if metric=='gower':
        bool_cat_cols = [col1 in cat_cols for col1 in a.columns]
        num_cols = [col2 for col2 in a.columns if col2 not in cat_cols]
        a[num_cols] = a[num_cols].astype("float")
        b[num_cols] = b[num_cols].astype("float")
        return gower_knn(a,b,bool_cat_cols)
    if metric=='euclid':
        return eucledian_knn(a,b)
    else: raise Exception("Unknown metric; options are 'gower' or 'euclid'")

# class nn_distance_metric():
#     def __init__(self, real, fake, cat_cols, metric='euclid'):
#         self.real = real
#         self.fake = fake
#         self.metric = metric

#         self.bool_cat_cols = [col1 in cat_cols for col1 in real.columns]

#         if (metric != 'gower' and metric != 'euclid'): 
#             raise Exception("Unknown metric; options are 'gower' or 'euclid'")
        
#         self.nn_rr = None
#         self.nn_rf = None
#         self.nn_fr = None
#         self.nn_ff = None
#         pass

#     def nn_real_real(self, weights=None):
#         """Calculates the distances to nearest neighbours in the real dataset"""
#         if (self.nn_rr==None or weights is not None):
#             d = _knn_distance(self.real,self.real,self.bool_cat_cols,2,self.metric,weights)
#             if weights is None:
#                 self.nn_rr = d[0], d[1]
#             return d[0], d[1]
#         else: return self.nn_rr

#     def nn_fake_fake(self, weights=None):
#         """Calculates the distances to nearest neighbours in the real dataset"""
#         if (self.nn_ff==None or weights is not None):
#             d = _knn_distance(self.fake,self.fake,self.bool_cat_cols,2,self.metric,weights)
#             if weights is None:
#                 self.nn_ff = d[0], d[1]
#             return d[0], d[1]
#         else: return self.nn_ff

#     def nn_real_fake(self, weights=None):
#         """Calculates the distances to nearest neighbours in the real dataset"""
#         if (self.nn_rf==None or weights is not None):
#             d = _knn_distance(self.real,self.fake,self.bool_cat_cols,2,self.metric,weights)
#             if weights is None:
#                 self.nn_rf = d[0], d[1]
#             return d[0], d[1]
#         else: return self.nn_rf

#     def nn_fake_real(self, weights=None):
#         """Calculates the distances to nearest neighbours in the real dataset"""
#         if (self.nn_fr==None or weights is not None):
#             d = _knn_distance(self.fake,self.real,self.bool_cat_cols,2,self.metric,weights)
#             if weights is None:
#                 self.nn_fr = d[0], d[1]
#             return d[0], d[1]
#         else: return self.nn_fr

    
