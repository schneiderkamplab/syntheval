# Description: Script for holding all the lesser functions
# Author: Anton D. Lautrup
# Date: 16-11-2022

import random
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, sem

from sklearn.decomposition import PCA
from scipy.stats import permutation_test, chi2_contingency, entropy
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score, f1_score

from .file_utils import stack
from .plt_utils import PlotDWM, PlotPCA, PlotCORR, PlotMI

### Functions for extras 
def scott_ref_rule(set1,set2):
    """Function for doing the Scott reference rule to calcualte number of bins needed to 
    represent the nummerical values."""
    samples = np.concatenate((set1, set2))
    std = np.std(samples)
    n = len(samples)
    bin_width = np.ceil(n**(1/3) * std / (3.5 * (np.percentile(samples, 75) - np.percentile(samples, 25)))).astype(int)

    min_edge = min(samples); max_edge = max(samples)
    N = min(abs(int((max_edge-min_edge)/bin_width)),10000); Nplus1 = N + 1
    return np.linspace(min_edge, max_edge, Nplus1)

### Functions to assist tests
def _pairwise_attributes_mutual_information(data):
    """Compute normalized mutual information for all pairwise attributes.

    Elements borrowed from: 
    Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017
    Presented at: Proceedingsof the 29th International Conference on Scientific and Statistical Database Management; 2017; Chicago.
    [doi:10.1145/3085504.3091117]"""

    labs = sorted(data.columns)
    res = (normalized_mutual_info_score(data[cat1].astype(str),data[cat2].astype(str),average_method='arithmetic') for cat1 in labs for cat2 in labs)
    return pd.DataFrame(np.fromiter(res, dtype=float).reshape(len(labs),len(labs)), columns = labs, index = labs)

def _knn_distance(a,b,cat_cols,metric='gower',weights=None):
    def eucledian_knn(a,b):
        """Function used for finding nearest neighbours"""
        nn = NearestNeighbors(n_neighbors=2,metric_params={'w':weights})
        if np.array_equal(a,b):
            nn.fit(a)
            d = nn.kneighbors(a)[0][:,1]
        else:
            nn.fit(b)
            d = nn.kneighbors(a)[0][:,0]
        return d

    def gower_knn(a,b,cat_cols):
        import gower
        """Function used for finding nearest neighbours"""
        if np.array_equal(a,b):
            d = gower.gower_matrix(a,cat_features=cat_cols,weight=weights)+np.eye(len(a))
            d = d.min(axis=1)
        else:
            d = gower.gower_matrix(a,b,cat_features=cat_cols,weight=weights)
            d = d.min(axis=1)
        return d

    if metric=='gower':
        return gower_knn(a,b,cat_cols)
    if metric=='euclid':
        return eucledian_knn(a,b)
    else: raise Exception("Unknown metric; options are 'gower' or 'euclid'")

def _adversarial_score(real,fake,cat_cols,metric):
    """Function for calculating adversarial score"""
    left = np.mean(_knn_distance(real, fake, cat_cols, metric) > _knn_distance(real, real, cat_cols, metric))
    right = np.mean(_knn_distance(fake, real, cat_cols, metric) > _knn_distance(fake, fake, cat_cols, metric))
    return 0.5 * (left + right)

def _hellinger(p,q):
    """Hellinger distance between distributions"""
    sqrt_pdf1 = np.sqrt(p)
    sqrt_pdf2 = np.sqrt(q)
    diff = sqrt_pdf1 - sqrt_pdf2
    return 1/np.sqrt(2)*np.sqrt(np.linalg.norm(diff))

def _discrete_ks_statistic(x, y):
    """Function for calculating the KS statistic"""
    KstestResult = ks_2samp(x,y)
    return np.round(KstestResult.statistic,4)

def _discrete_ks(x, y):
    """Function for doing permutation test of discrete values in the KS test"""
    res = permutation_test((x, y), _discrete_ks_statistic, n_resamples=1000, vectorized=False, permutation_type='independent', alternative='greater')
    # plt.figure()
    # plt.hist(res.null_distribution, bins=50)
    # plt.axvline(x=res.statistic, color='red', linestyle='--')
    # plt.savefig('permutation_test')
    # plt.close()
    return res.statistic, res.pvalue

def _cramers_V(var1,var2) :
    """function for calculating Cramers V between two categorial variables
    credit: https://www.kaggle.com/code/chrisbss1/cramer-s-v-correlation-matrix
    """
    crosstab =np.array(pd.crosstab(var1, var2, rownames=None, colnames=None)) # Cross table building
    stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
    return (stat/(obs*mini+1e-16))

def _apply_mat(data,func,labs1,labs2):
    """Help function for constructing a matrix based on func accross labels 1 and 2"""
    res = (func(data[lab1],data[lab2]) for lab1 in labs1 for lab2 in labs2)
    return pd.DataFrame(np.fromiter(res, dtype=float).reshape(len(labs1),len(labs2)), columns = labs2, index = labs1)

def _correlation_ratio(categories, measurements):
    """Function for calculating the correlation ration eta^2 of categorial and nummerical data"""
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

def mixed_correlation(data,num_cols,cat_cols):
    """Function for calculating a correlation matrix of mixed datatypes.
    Spearman's rho is used for rank-based correlation, Cramer's V is used for categorical variables, 
    and correlation ratio is used for categorical and continuous variables.
    """
    corr_num_num = data[num_cols].corr()
    corr_cat_cat = _apply_mat(data,_cramers_V,cat_cols,cat_cols)
    corr_cat_num = _apply_mat(data,_correlation_ratio,cat_cols,num_cols)

    top_row = pd.concat([corr_cat_cat,corr_cat_num],axis=1)
    bot_row = pd.concat([corr_cat_num.transpose(),corr_num_num],axis=1)
    corr = pd.concat([top_row,bot_row],axis=0)
    return corr

def class_test(real_models, fake_models, real, fake, test, F1_type='micro'):
    """Function for running a training session and getting predictions 
    on the SciPy model provided, and data."""
    res = []
    for r_mod, f_mod in zip(real_models, fake_models):
        r_mod.fit(real[0],real[1])
        f_mod.fit(fake[0],fake[1])

        pred_real = r_mod.predict(test[0])
        pred_fake = f_mod.predict(test[0])

        f1_real = f1_score(test[1],pred_real,average=F1_type)
        f1_fake = f1_score(test[1],pred_fake,average=F1_type)

        res.append([f1_real, f1_fake])
    return np.array(res).T

### Functions for the tests themselves
def dimensionwise_means(real, fake, num_cols, fig_index):
    """Function for calculating DWM, plotting an appropriate diagram"""
    r_scaled = MinMaxScaler().fit_transform(real[num_cols])
    f_scaled = MinMaxScaler().fit_transform(fake[num_cols])
    dim_means = np.array([np.mean(r_scaled,axis=0),np.mean(f_scaled,axis=0)]).T
    dim_sem = np.array([sem(r_scaled),sem(f_scaled)]).T
    PlotDWM(dim_means, dim_sem, num_cols, fig_index)
    pass

def principal_component_analysis(real,fake,num_cols,target, fig_index):
    r_scaled = StandardScaler().fit_transform(real[num_cols])
    f_scaled = StandardScaler().fit_transform(fake[num_cols])

    pca = PCA(n_components=2)
    r_pca = pca.fit_transform(r_scaled)
    f_pca = pca.transform(f_scaled)

    r_pca = pd.DataFrame(r_pca,columns=['PC1', 'PC2'])
    f_pca = pd.DataFrame(f_pca,columns=['PC1', 'PC2'])
    r_pca['target'] = real[target]
    f_pca['target'] = fake[target]
    PlotPCA(r_pca,f_pca, fig_index)
    pass

def correlation_matrix_difference(real, fake, num_cols, cat_cols, mixed=True):
    """Function for calculating the (mixed) correlation matrix difference.
    This calculation uses spearmans rho for numerical-numerical, Cramer's V for categories,
    and correlation ratio (eta) for numerical-categorials.
    
    Mixed mode can be disabled, to only use the numerical variables."""

    if mixed==True:
        r_corr = mixed_correlation(real,num_cols,cat_cols)
        f_corr = mixed_correlation(fake,num_cols,cat_cols)
    else:
        r_corr = real[num_cols].corr()
        f_corr = fake[num_cols].corr()
    
    # fig, axs = plt.subplots(1,2,figsize=(12,4))
    # sns.heatmap(r_corr,annot=True, fmt='.2f',ax=axs[0])
    # sns.heatmap(f_corr,annot=True, fmt='.2f',ax=axs[1])
    # plt.show()

    corr_mat = r_corr-f_corr
    PlotCORR(corr_mat)
    return np.linalg.norm(corr_mat,ord='fro')

def mutual_information_matrix_difference(real,fake):
    """Function for calculating the mutual information matrix difference"""
    r_mi = _pairwise_attributes_mutual_information(real)
    f_mi = _pairwise_attributes_mutual_information(fake)

    mi_mat = r_mi - f_mi
    PlotMI(mi_mat)
    return np.linalg.norm(mi_mat,ord='fro')

def featurewise_ks_test(real, fake, cat_cols, sig_lvl=0.05):
    """Function for executing the Kolmogorov-Smirnov test.
    
    Returns:
        Avg. KS dist: dict  - holds avg. and standard error of the mean (SE)
        Avg. KS pval: dict  - holds avg. and SE
        num of sigs : int   - the number of significant tests at sig_lvl
        frac of sigs: float - the fraction of significant tests at sig_lvl   
     """
    dists = []
    pvals = []

    for category in real.columns:
        R = real[category]
        F = fake[category]

        if category in cat_cols:
            statistic, pvalue = _discrete_ks(F,R)
            dists.append(statistic)
            pvals.append(pvalue)
        else:
            KstestResult = ks_2samp(R,F)
            dists.append(KstestResult.statistic)
            pvals.append(KstestResult.pvalue)

    ### Calculate number of significant tests, and fraction of sifnificant tests
    num  = sum([p_val < sig_lvl for p_val in pvals])
    frac = num/len(pvals)

    return {'avg': np.mean(dists), 'err': np.std(dists,ddof=1)/np.sqrt(len(dists))}, {'avg': np.mean(pvals), 'err':np.std(pvals,ddof=1)/np.sqrt(len(pvals))}, num, frac 

def featurewise_hellinger_distance(real, fake, cat_cols, num_cols):
    """Function for calculating the hellinger distance of the categorial 
    and nummerical features"""
    H_dist = []
    
    for category in cat_cols:
        class_num = len(np.unique(real[category]))

        pdfR = np.histogram(real[category], bins=class_num, density=True)[0]
        pdfF = np.histogram(fake[category], bins=class_num, density=True)[0]
        H_dist.append(_hellinger(pdfR,pdfF))
    
    for category in num_cols:
        #n_bins = int(2*(len(real[category])**(1/3))) # Rice rule for determining bin number
        n_bins = scott_ref_rule(real[category],fake[category]) # Scott rule for finding bin width

        pdfR = np.histogram(real[category], bins=n_bins, density=True)[0]
        pdfF = np.histogram(fake[category], bins=n_bins, density=True)[0]

        H_dist.append(_hellinger(pdfR,pdfF))

    return {'avg': np.mean(H_dist), 'err': np.std(H_dist,ddof=1)/np.sqrt(len(H_dist))}

def confidence_interval_overlap(real, fake, num_cols):
    """Function for calculating the average CIO, also returns the 
    number of nonoverlapping interval"""
    mus = np.array([np.mean(real[num_cols],axis=0),np.mean(fake[num_cols],axis=0)]).T
    sems = np.array([sem(real[num_cols]),sem(fake[num_cols])]).T
    
    CI = sems*1.96
    us = mus+sems
    ls = mus-sems

    Jk = []
    for i in range(len(CI)):
        top = (min(us[i][0],us[i][1])-max(ls[i][0],ls[i][1]))
        Jk.append(max(0,0.5*(top/(us[i][0]-ls[i][0])+top/(us[i][1]-ls[i][1]))))

    num = sum([j == 0 for j in Jk])
    frac = num/len(Jk)

    return {'avg': np.mean(Jk), 'err': np.std(Jk,ddof=1)/np.sqrt(len(Jk))}, num, frac

def propensity_mean_square_error(real, fake):
    """Train a a discriminator to distinguish between real and fake data."""
    discriminator = MLPClassifier(random_state=42)
    Df = stack(real,fake).drop(['index'], axis=1)

    Xs, ys = Df.drop(['real'], axis=1), Df['real']

    ### Run 5-fold cross-validation
    kf = KFold(n_splits=5)
    res, acc = [], []
    for train_index, test_index in kf.split(Df):
        x_train = Xs.iloc[train_index]
        x_test = Xs.iloc[test_index]
        y_train = ys.iloc[train_index]
        y_test = ys.iloc[test_index]

        mod = discriminator.fit(x_train,y_train)
        pred = mod.predict_proba(x_test)
        
        res.append(np.mean((pred[:,0]-0.5)**2))
        acc.append(f1_score(y_test,mod.predict(x_test),average='macro'))

    return {'avg': np.mean(res), 'err': np.std(res,ddof=1)/np.sqrt(len(res))},  {'avg': np.mean(acc), 'err': np.std(acc,ddof=1)/np.sqrt(len(acc))}

def adversarial_accuracy(real, fake, cat_cols, num_cols, metric, n_batches=30):
    """Implementation heavily inspired by original paper"""
    bool_cat_cols = [col1 in cat_cols for col1 in real.columns]

    real[num_cols] = MinMaxScaler().fit_transform(real[num_cols])
    fake[num_cols] = MinMaxScaler().fit_transform(fake[num_cols])
    
    if len(real)*2 < len(fake):
        aa_lst = []
        for batch in range(n_batches):
            temp_f = fake.sample(n=len(real))
            aa_lst.append(_adversarial_score(real,temp_f,bool_cat_cols, metric))
        return {'avg': np.mean(aa_lst), 'err': np.std(aa_lst,ddof=1)/np.sqrt(len(aa_lst))}
    else:
        return {'avg': _adversarial_score(real,fake,bool_cat_cols,metric), 'err': 0.0}

def distance_to_closest_record(real,fake,cat_cols,num_cols,metric):
    """Distance to closest record, using the same NN stuff as NNAA"""
    bool_cat_cols = [col1 in cat_cols for col1 in real.columns]
    distances = _knn_distance(fake,real,bool_cat_cols,metric)
    in_dists = _knn_distance(real,real,bool_cat_cols,metric)

    # int_nn_avg = np.mean(in_dists)#np.median(in_dists)
    # int_nn_err = np.std(in_dists,ddof=1)/np.sqrt(len(in_dists))
    # min_dist_avg = np.mean(distances)#np.median(distances)
    # min_dist_err = np.std(distances,ddof=1)/np.sqrt(len(distances))

    int_nn = np.median(in_dists)
    mut_nn = np.median(distances)

    #dcr = min_dist_avg/int_nn_avg
    #dcr_err = np.sqrt((min_dist_err/min_dist_avg)**2+(int_nn_err/int_nn_avg)**2)
    dcr = mut_nn/int_nn
    return dcr#{'avg': dcr, 'err': dcr_err}

def nearest_neighbour_distance_ratio(real, fake, num_cols):
    """
    Compute the Nearest Neighbour Distance Ratio (NNDR) between two datasets.
    """
    nbrs = NearestNeighbors(n_neighbors=2).fit(real[num_cols])
    distance, _ = nbrs.kneighbors(fake[num_cols])
    dr = list(map(lambda x: x[0] / x[1], distance))
    return {'avg': np.mean(dr), 'err': np.std(dr,ddof=1)/np.sqrt(len(dr))}

def hitting_rate(real,fake,cat_cols):
    """For hitting rate we regard records as similar if the 
    nummerical attributes are within a threshold range(att)/30"""
    thres = (real.max() - real.min())/30
    thres[cat_cols] = 0

    hit = 0
    for i, r in real.iterrows():
        hit += any((abs(r-fake) <= thres).all(axis='columns'))
    hit_rate = hit/len(real)
    return hit_rate

def epsilon_identifiability(real, fake, num_cols, cat_cols, metric):
    """Function for computing the epsilon identifiability risk

    Adapted from:
    Yoon, J., Drumright, L. N., & van der Schaar, M. (2020). Anonymization Through Data Synthesis Using Generative Adversarial Networks (ADS-GAN). 
    IEEE Journal of Biomedical and Health Informatics, 24(8), 2378–2388. [doi:10.1109/JBHI.2020.2980262] 
    """

    # Entropy computation
    def column_entropy(labels):
        value, counts = np.unique(np.round(labels), return_counts=True)
        return entropy(counts)

    bool_cat_cols = [col1 in cat_cols for col1 in real.columns]

    if metric == 'euclid':
        real = np.asarray(real[num_cols])
        fake = np.asarray(fake[num_cols])
    else: 
        real, fake = np.asarray(real), np.asarray(fake)

    no, x_dim = np.shape(real)
    W = [column_entropy(real[:, i]) for i in range(x_dim)]
    W_adjust = 1/(np.array(W)+1e-16)

    # for i in range(x_dim):
    #     real_hat[:, i] = real[:, i] * 1. / W[i]
    #     fake_hat[:, i] = fake[:, i] * 1. / W[i]

    in_dists = _knn_distance(real,real,bool_cat_cols,metric,W_adjust)
    ext_distances = _knn_distance(real,fake,bool_cat_cols,metric,W_adjust)

    R_Diff = ext_distances - in_dists
    identifiability_value = np.sum(R_Diff < 0) / float(no)

    return identifiability_value