# Description: Script for holding plotting functionalities of the table evaluator pro
# Author: Anton D. Lautrup
# Date: 01-02-2023

import time
import math

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from syntheval.utils.preprocessing import stack

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
        plt.errorbar(m_diff,range(len(m_diff)),xerr=np.array(pr_sem)*1.96,marker='o',linestyle='none', capsize=6, markersize="6")
        labels = [label[:10] + '...' if len(label) > 10 else label for label in labels]
        plt.yticks(range(len(m_diff)), labels)
        plt.vlines(0,-0.5,len(means)-0.5,colors='k',alpha=0.5)
        
        plt.title(r"Dimensionwise means (95% confidence intervals)")
        plt.xlabel('mean difference')
        plt.tight_layout()
        plt.grid(linestyle='--', alpha=0.5)
        plt.savefig('SE_dwm_' +str(int(time.time()))+'.png')
    else:
        y = lambda x, a : a*x
        popt, pcov = curve_fit(y, means[:,0], means[:,1])
        xline = np.linspace(min(means[:,0])-0.01, max(means[:,0])+0.01, 10)

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
    pass

def plot_principal_components(reals, fakes):
    """Plot PCA components of real and synthetic data i a pairplot"""
    class_num = len(np.unique(reals['target']))
    components = [col for col in reals.columns if col != 'target']
    comp_num = len(components)

    if comp_num < 2:
        print('Error: Principal component analysis - too few components to plot!')
    elif comp_num == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True, sharex=True)
        sns.scatterplot(x=reals[components[0]], y=reals[components[1]], hue=reals['target'], ax=ax1, palette=sns.color_palette("colorblind",class_num))
        sns.scatterplot(x=fakes[components[0]], y=fakes[components[1]], hue=fakes['target'], ax=ax2, palette=sns.color_palette("colorblind",class_num))
        ax1.set_title('real data'),ax1.legend().remove()
        ax2.set_title('synthetic data'),ax2.legend().remove()
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, title = 'class', loc='center right')
        fig.tight_layout()
        fig.subplots_adjust(right=0.85)
        plt.savefig('SE_pca_proj_' +str(int(time.time()))+'.png')
    else:
        fig, axs = plt.subplots(comp_num, comp_num, figsize=(comp_num*3, comp_num*3), sharey=True, sharex=True)
        plt.suptitle("Synthetic (U) and real data (L) projected onto real data PCA components",fontsize=14)
        for i in range(comp_num):
            for j in range(comp_num):
                # Only plot off-diagonal elements
                if i != j:
                    # Upper triangle: use data from df1
                    if i < j:
                        sns.scatterplot(x=fakes[components[j]], y=fakes[components[i]], hue=fakes['target'], ax=axs[i, j], palette=sns.color_palette("colorblind",class_num))
                    # Lower triangle: use data from df2
                    else:
                        sns.scatterplot(x=reals[components[j]], y=reals[components[i]], hue=reals['target'], ax=axs[i, j], palette=sns.color_palette("colorblind",class_num))
                    axs[i, j].legend().remove()
                elif i == 0:
                    # Hide the diagonal plots
                    axs[i, j].set_ylabel(components[i])
                elif i == comp_num and j== comp_num:
                    axs[i, j].set_xlabel(components[j])

        # Create a single legend for both subplots
        handles, labels = axs[1,2].get_legend_handles_labels()
        fig.legend(handles, labels, title = 'class', loc='center')
        fig.tight_layout()
        plt.savefig('SE_pca_proj_' +str(int(time.time()))+'.png')
        plt.close()
    pass

def plot_own_principal_component_pairplot(data):
    components = [col for col in data.columns if col not in ['index', 'target', 'real']]
    size = len(components)

    df_real = data[data['real'] == 1]
    df_fake = data[data['real'] == 0]

    fig, axs = plt.subplots(size, size, figsize=(size*3, size*3))
    if size < 3: plt.suptitle("Synthetic (U) and real (L) data projected onto own PCA components",fontsize=12)
    else: plt.suptitle("Synthetic (U) and real (L) data projected onto own PCA components",fontsize=14)
    for i in range(size):
        for j in range(size):
            if i != j:
                if i < j: # Upper triangle: use data from df_fake
                    sns.scatterplot(x=df_fake[components[j]], y=df_fake[components[i]], ax=axs[i, j], c=['#7FB8D8'],edgecolor='k')
                else:     # Lower triangle: use data from df_real
                    sns.scatterplot(x=df_real[components[j]], y=df_real[components[i]], ax=axs[i, j], c=['#EEC681'],edgecolor='k')
            else:
                sns.kdeplot(x=data[components[i]], hue=data['real'], fill=True, multiple="layer", ax=axs[i, i], palette=['#7FB8D8','#EEC681'])

    fig.tight_layout()
    plt.savefig('SE_pca_own_' +str(int(time.time()))+'.png')
    plt.close()
    pass

def plot_significantly_dissimilar_variables(real, fake, labels, cat_cols):
    """Plot histograms of every attribute that is significantly unlike the one it is modelled on"""

    df = stack(real, fake)

    plt.rc('font', size=6)
    fig, axes = plt.subplots(nrows=math.ceil(len(labels)/4), ncols=4, figsize=(10, 2*math.ceil(len(labels)/4)))
    axes = axes.flatten()

    for i, column in enumerate(labels):
        
        if column in cat_cols: sns.histplot(data=df, x=column, hue='real', stat='probability', common_norm=False, discrete=True, multiple="dodge", alpha=0.5, shrink=.8, ax=axes[i])
        else: sns.histplot(data=df, x=column, hue='real', stat='probability', common_norm=False, multiple="layer", alpha=0.5, shrink=.8, ax=axes[i])
        
        axes[i].set_title(f'Variable {column}',fontsize=8)
        
    plt.tight_layout()
    plt.savefig('SE_sig_hists_' +str(int(time.time()))+ '.png')
    plt.close()
    pass

def _shortened_labels(ax_get_ticks):
    max_label_length = 10
    labels = [label.get_text()[:max_label_length] + '...' if len(label.get_text()) > max_label_length else label.get_text() for label in ax_get_ticks]
    return labels

def plot_matrix_heatmap(mat,title,file_name,axs_lim,axs_scale):
    """Plotting difference matrix heatmap"""
    s = max(8,int(np.shape(mat)[0]/3))
    fig, ax = plt.subplots(figsize=(s,s))
    if s <= 8: sns.heatmap(mat, annot=True, fmt='.2f', cmap=axs_scale, ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))
    else: sns.heatmap(mat, cmap=axs_scale, ax=ax, cbar=True, mask=np.triu(np.ones(mat.shape), k=1))
    if axs_scale is not None: ax.collections[0].set_clim(axs_lim) 

    plt.title(title)
    labels = _shortened_labels(ax.get_xticklabels())
    ax.set_xticks(ax.get_xticks(), labels, rotation=35, ha='right')
    ax.set_yticks(ax.get_yticks(), labels)
    fig.tight_layout()
    plt.savefig('SE_' +file_name +'_' +str(int(time.time()))+ '.png')

    pass

def plot_roc_curves(real_roc_mean, real_roc_conf, fake_roc, fake_roc_conf, title, file_name):
    fpr1, tpr1, roc_auc1 = real_roc_mean[0], real_roc_mean[1], real_roc_mean[2]
    mean_fpr_real, mean_tpr_real, std_tpr_real = real_roc_conf[0], real_roc_conf[1], real_roc_conf[2]
    fpr2, tpr2, roc_auc2 = fake_roc[0], fake_roc[1], fake_roc[2]
    mean_fpr_fake, mean_tpr_fake, std_tpr_fake = fake_roc_conf[0], fake_roc_conf[1], fake_roc_conf[2]
    plt.figure(figsize=(6, 6))
    plt.fill_between(mean_fpr_real, mean_tpr_real - 1.96*std_tpr_real, mean_tpr_real + 1.96*std_tpr_real, color='lightblue', alpha=0.5)
    plt.fill_between(mean_fpr_fake, mean_tpr_fake - 1.96*std_tpr_fake, mean_tpr_fake + 1.96*std_tpr_fake, color='lightpink', alpha=0.5)
    plt.plot(fpr1, tpr1, color='blue', lw=0.5, label=f'real data (AUROC = {roc_auc1:.4f})')
    plt.plot(fpr2, tpr2, color='red', lw=0.5, label=f'synt data (AUROC = {roc_auc2:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {title} models')
    plt.legend(loc='lower right')
    plt.savefig('SE_' + file_name +'_' +str(int(time.time()))+ '.png')
    
    pass