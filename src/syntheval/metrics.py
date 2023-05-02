# Description: Script for holding all the lesser functions
# Author: Anton D. Lautrup
# Date: 16-11-2022

import random
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, sem

from sklearn.decomposition import PCA
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
def _pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes. Return a DataFrame.
        Implementation stolen from 
        
        Ping H, Stoyanovich J, Howe B. DataSynthesizer: privacy-preserving synthetic datasets. 2017 
        Presented at: Proceedingsof the 29th International Conference on Scientific and Statistical Database Management; 2017; Chicago. 
        [doi:10.1145/3085504.3091117]
    
    """
    sorted_columns = sorted(dataset.columns)
    mi_df = pd.DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str),
                                                               average_method='arithmetic')
    return mi_df

def _distance(a,b):
    """Function used for finding nearest neighbours"""
    nn = NearestNeighbors(n_neighbors=2)
    if np.array_equal(a,b):
        nn.fit(a)
        d = nn.kneighbors()[0][:,1]
    else:
        nn.fit(b)
        d = nn.kneighbors(a)[0][:,0]
    return d

def _adversarial_score(real,fake):
    """Function for calculating adversarial score"""
    left = np.mean(_distance(real,fake) > _distance(real, real))
    right = np.mean(_distance(fake,real) > _distance(fake, fake))
    return 0.5 * (left + right)

def _hellinger(p,q):
    """Hellinger distance between distributions"""
    sqrt_pdf1 = np.sqrt(p)
    sqrt_pdf2 = np.sqrt(q)
    diff = sqrt_pdf1 - sqrt_pdf2
    return np.sqrt(np.linalg.norm(diff))

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
    return np.array(res)

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

#TODO: make this work for categorials too <ADL 09-03-2023 11:56>
def correlation_matrix_difference(real,fake,num_cols):
    """Function for calculating the correlation matrix difference, 
    but only for the nummerical columns"""
    r_corr = real[num_cols].corr()
    f_corr = fake[num_cols].corr()

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

def featurewise_ks_test(real,fake, sig_lvl=0.05):
    """Function for running the KS test in each of the marginals, returns
    the average p-value and the number of significant tests."""
    dists = []
    pvals = []
    for category in real.columns:
        R = real[category]
        F = fake[category]
        stat, pval = ks_2samp(R,F)
        dists.append(stat)
        pvals.append(pval)

    ### Calculate number of significant tests, and fraction of sifnificant tests
    num  = sum([p_val < sig_lvl for p_val in pvals])
    frac = num/len(pvals)
    
    return np.mean(dists), np.mean(pvals), num, frac 

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

    return np.mean(H_dist)

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

    return np.mean(Jk), sum([j == 0 for j in Jk])

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
        acc.append(f1_score(y_test,mod.predict(x_test),average='micro'))

    return np.mean(res), np.mean(acc)

def adversarial_accuracy(real, fake, n_batches=30):
    """Implementation heavily inspired by original paper"""
    r_scaled = MinMaxScaler().fit_transform(real)
    f_scaled = MinMaxScaler().fit_transform(fake)

    if len(r_scaled)*2 < len(f_scaled):
        aa_lst = []
        for batch in range(n_batches):
            temp_f = random.choices(f_scaled, k=len(r_scaled))#fake.sample(len(real))
            
            aa_lst.append(_adversarial_score(r_scaled,temp_f))
        return np.mean(aa_lst)
    else:
        return _adversarial_score(r_scaled,f_scaled)

def distance_to_closest_record(real,fake):
    """Distance to closest record, using the same NN stuff as NNAA"""
    distances = _distance(real,fake)
    in_dists = _distance(real,real)

    int_nn_avg = np.mean(in_dists)

    min_distances = np.mean(distances)
    return np.mean(min_distances)/int_nn_avg

def hitting_rate(real,fake,cat_cols):
    """For hitting rate we regard records as similar if the 
    nummerical attributes are within a threshold range(att)/30"""
    thres = (real.max()- real.min())/30
    thres[cat_cols] = 0

    hit = 0
    for i, r in real.iterrows():
        hit += any((abs(r-fake) <= thres).all(axis='columns'))
    hit_rate = hit/len(real)
    return hit_rate

