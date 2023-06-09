# Description: Script for holding plotting functionalities of the table evaluator pro
# Author: Anton D. Lautrup
# Date: 01-02-2023

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# params = {'text.usetex' : True,
#           'font.size' : 14,
#           'font.family' : 'lmodern'
#           }
# plt.rcParams.update(params) 

def PlotDWM(means,sem,labels,fig_index):
    """Plot the dimensionwise means of real and synthetic data and note down the GoF"""

    if len(means) < 10:
        m_diff = means[:,0]-means[:,1]
        pr_sem = np.sum(sem,axis=1)
        fig, ax = plt.subplots(figsize=(7,5))
        #plt.scatter(m_diff,range(len(m_diff)))
        plt.errorbar(m_diff,range(len(m_diff)),xerr=np.array(pr_sem)*1.96,marker='o',linestyle='none', capsize=6, markersize="6")
        plt.yticks(range(len(m_diff)), labels)
        plt.vlines(0,-0.5,len(means)-0.5,colors='k',alpha=0.5)
        
        plt.title(r"\begin{center} Differences in means \\ \normalsize{}{95\% confidence intervals} \end{center}")
        plt.xlabel('mean difference')
        plt.tight_layout()
        plt.grid(linestyle='--', alpha=0.5)
        plt.savefig('plot_dwm_'+str(fig_index)+'.png')
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

        plt.title(r"\begin{center} Dimensionwise means \\ \normalsize{}{95\% confidence intervals} \end{center}")
        plt.xlabel('real data')
        plt.ylabel('synthetic data')
        ax.text(0.95, 0.01, ('CC = ' + str(np.round(popt[0],3))),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,fontsize=15)

        plt.tight_layout()
        plt.grid(linestyle='--', alpha=0.3)
        plt.savefig('plot_dwm_'+str(fig_index)+'.png')
        #plt.show()
    pass

def PlotPCA(reals, fakes,fig_index):
    """Plot first two PCA components of real and synthetic data side by side"""
    class_num = len(np.unique(reals['target']))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    sns.scatterplot(x=reals['PC1'], y=reals['PC2'], hue=reals['target'], ax=ax1, palette=sns.color_palette("colorblind",class_num))
    sns.scatterplot(x=fakes['PC1'], y=fakes['PC2'], hue=fakes['target'], ax=ax2, palette=sns.color_palette("colorblind",class_num))

    ax1.set_title('real data'),ax1.legend().remove()
    ax2.set_title('fake data'),ax2.legend().remove()

    # Create a single legend for both subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, title = 'class', loc='center right')
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    plt.savefig('plot_pca_'+str(fig_index)+'.png')
    #plt.show()
    pass

def PlotCORR(mat):
    """Plotting the correlation difference matrix"""
    # Plotting
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(mat, annot=True,fmt='.2f', cmap='RdBu', ax=ax, cbar=True,mask=np.triu(np.ones(mat.shape), k=1))

    # Add labels to the plot
    plt.title(r"\begin{center} Correlation matrix difference \\ \normalsize{}{nummerical values only} \end{center}")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig('plot_corr.png')
    #plt.show()
    pass

def PlotMI(mat):
    """Plotting the pairwise mutual information matrix difference"""
    # Plotting
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(mat, annot=True,fmt='.2f', cmap='RdBu', ax=ax, cbar=True,mask=np.triu(np.ones(mat.shape), k=1))

    # Add labels to the plot
    plt.title(r"\begin{center} Mutual information matrix difference \\ \normalsize{}{DataSynthesizer version, all datatypes} \end{center}")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    plt.savefig('plot_mi.png')
    #plt.show()
    pass


