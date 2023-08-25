# Description: Script for holding plotting functionalities of the table evaluator pro
# Author: Anton D. Lautrup
# Date: 01-02-2023

import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# params = {'text.usetex' : True,
#           'font.size' : 14,
#           'font.family' : 'lmodern'
#           }
# plt.rcParams.update(params) 

def plot_dimensionwise_means(means, sem, labels):
    """Plot the dimensionwise means of real and synthetic data and note down the GoF"""

    if len(means) < 10:
        m_diff = means[:,0]-means[:,1]
        pr_sem = np.sqrt(np.sum(sem**2,axis=1))
        fig, ax = plt.subplots(figsize=(6,5))
        #plt.scatter(m_diff,range(len(m_diff)))
        plt.errorbar(m_diff,range(len(m_diff)),xerr=np.array(pr_sem)*1.96,marker='o',linestyle='none', capsize=6, markersize="6")
        labels = [label[:10] + '...' if len(label) > 10 else label for label in labels]
        plt.yticks(range(len(m_diff)), labels)
        plt.vlines(0,-0.5,len(means)-0.5,colors='k',alpha=0.5)
        
        plt.title(r"Dimensionwise means (95% confidence intervals)")
        plt.xlabel('mean difference')
        plt.tight_layout()
        plt.grid(linestyle='--', alpha=0.5)
        plt.savefig('SE_dwm_' +str(int(time.time()))+'.png')
        #plt.show()
    else:
        y = lambda x, a : a*x
        popt, pcov = curve_fit(y, means[:,0], means[:,1])
        xline = np.linspace(min(means[:,0])-0.01, max(means[:,0])+0.01, 10)
        #print(popt,pcov)

        fig, ax = plt.subplots(figsize=(5,5))
        
        plt.errorbar(means[:,0],means[:,1],xerr=np.array(sem[:,0])*1.96,yerr=np.array(sem[:,1])*1.96,
                            marker='o',linestyle='none', capsize=2, markersize="2")
        plt.plot(xline,y(xline,1))

        plt.title(r"Dimensionwise means (95% confidence intervals)")
        plt.xlabel('real data')
        plt.ylabel('synthetic data')
        ax.text(0.95, 0.01, ('CC = ' + str(np.round(popt[0],3))),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,fontsize=15)

        plt.tight_layout()
        plt.grid(linestyle='--', alpha=0.3)
        plt.savefig('SE_dwm_' +str(int(time.time()))+'.png')
        #plt.show()
    pass

def plot_principal_components(reals, fakes):
    """Plot first two PCA components of real and synthetic data side by side"""
    class_num = len(np.unique(reals['target']))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    sns.scatterplot(x=reals['PC1'], y=reals['PC2'], hue=reals['target'], ax=ax1, palette=sns.color_palette("colorblind",class_num))
    sns.scatterplot(x=fakes['PC1'], y=fakes['PC2'], hue=fakes['target'], ax=ax2, palette=sns.color_palette("colorblind",class_num))

    ax1.set_title('real data'),ax1.legend().remove()
    ax2.set_title('synthetic data'),ax2.legend().remove()

    # Create a single legend for both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title = 'class', loc='center right')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig('SE_pca_' +str(int(time.time()))+'.png')
    #plt.show()
    pass

def _shortened_labels(ax_get_ticks):
    max_label_length = 10
    labels = [label.get_text()[:max_label_length] + '...' if len(label.get_text()) > max_label_length else label.get_text() for label in ax_get_ticks]
    return labels

def plot_matrix_heatmap(mat,title,file_name):
    """Plotting difference matrix heatmap"""
    s = max(8,int(np.shape(mat)[0]/3))
    fig, ax = plt.subplots(figsize=(s,s))
    if s <= 8: sns.heatmap(mat, annot=True, fmt='.2f', cmap='RdBu', ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))
    else: sns.heatmap(mat, cmap='RdBu', ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))

    plt.title(title)
    labels = _shortened_labels(ax.get_xticklabels())
    ax.set_xticks(ax.get_xticks(), labels, rotation=45, ha='right')
    ax.set_yticks(ax.get_yticks(), labels)
    fig.tight_layout()
    plt.savefig('SE_' +file_name +'_' +str(int(time.time()))+ '.png')

    pass

# def plot_correlation(mat):
#     """Plotting the correlation difference matrix"""
#     # Plotting
#     s = max(8,int(np.shape(mat)[0]/3))
#     fig, ax = plt.subplots(figsize=(s,s))
#     if s <= 8: sns.heatmap(mat, annot=True, fmt='.2f', cmap='RdBu', ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))
#     else: sns.heatmap(mat, cmap='RdBu', ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))

#     plt.title("Correlation matrix difference")
#     labels = _shortened_labels(ax.get_xticklabels())
#     ax.set_xticks(ax.get_xticks(), labels, rotation=45, ha='right')
#     ax.set_yticks(ax.get_yticks(), labels)
#     fig.tight_layout()
#     plt.savefig('plot_corr.png')
#     #plt.show()
#     pass

# def plot_mutual_information(mat):
#     """Plotting the pairwise mutual information matrix difference"""
#     # Plotting
#     s = max(8,int(np.shape(mat)[0]/3))
#     fig, ax = plt.subplots(figsize=(s,s))
#     if s <= 8: sns.heatmap(mat, annot=True, fmt='.2f', cmap='RdBu', ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))
#     else: sns.heatmap(mat, cmap='RdBu', ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))

#     plt.title("Mutual information matrix difference")
#     labels = _shortened_labels(ax.get_xticklabels())
#     ax.set_xticks(ax.get_xticks(), labels, rotation=45, ha='right')
#     ax.set_yticks(ax.get_yticks(), labels)
#     fig.tight_layout()
#     plt.savefig('plot_mi.png')
#     #plt.show()
#     pass


