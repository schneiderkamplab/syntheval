# Description: Hellinger distance metric class
# Author: Anton D. Lautrup
# Date: 18-08-2023

import numpy as np

from ..core.metric import MetricClass

def _scott_ref_rule(set1,set2):
    """Function for doing the Scott reference rule to calcualte number of bins needed to 
    represent the nummerical values."""
    samples = np.concatenate((set1, set2))
    std = np.std(samples)
    n = len(samples)
    bin_width = np.ceil(n**(1/3) * std / (3.5 * (np.percentile(samples, 75) - np.percentile(samples, 25)))).astype(int)

    min_edge = min(samples); max_edge = max(samples)
    N = min(abs(int((max_edge-min_edge)/bin_width)),10000); Nplus1 = N + 1
    return np.linspace(min_edge, max_edge, Nplus1)

def _hellinger(p,q):
    """Hellinger distance between distributions"""
    sqrt_pdf1 = np.sqrt(p)
    sqrt_pdf2 = np.sqrt(q)
    diff = sqrt_pdf1 - sqrt_pdf2
    return 1/np.sqrt(2)*np.sqrt(np.linalg.norm(diff))

class HellingerDistance(MetricClass):

    def name() -> str:
        """name/keyword to reference the metric"""
        return 'h_dist'

    def type() -> str:
        """privacy or utility"""
        return 'utility'

    def evaluate(self) -> float | dict:
        """ Function for evaluating the metric"""
        H_dist = []
    
        for category in self.cat_cols:
            class_num = len(np.unique(self.real_data[category]))

            pdfR = np.histogram(self.real_data[category], bins=class_num, density=True)[0]
            pdfF = np.histogram(self.synt_data[category], bins=class_num, density=True)[0]
            H_dist.append(_hellinger(pdfR,pdfF))
        
        for category in self.num_cols:
            n_bins = _scott_ref_rule(self.real_data[category],self.synt_data[category]) # Scott rule for finding bin width

            pdfR = np.histogram(self.real_data[category], bins=n_bins, density=True)[0]
            pdfF = np.histogram(self.synt_data[category], bins=n_bins, density=True)[0]
            H_dist.append(_hellinger(pdfR,pdfF))

        self.results = {'avg': np.mean(H_dist), 'err': np.std(H_dist,ddof=1)/np.sqrt(len(H_dist))}
        return self.results

    def format_output(self) -> str:
        """ Return string for formatting the output, when the
        metric is part of SynthEval.        
        """
        string = """\
| Average empirical Hellinger distance     :   %.4f  %.4f   |""" % (self.results['avg'], self.results['err'])
        return string

    def normalize_output(self) -> list:
        """ This function is for making a dictionary of the most quintessential
        nummerical results of running this metric (to be turned into a dataframe).

        The required format is:
        metric  dim  val  err  n_val  n_err
            name1  u  0.0  0.0    0.0    0.0
            name2  p  0.0  0.0    0.0    0.0
        """
        if self.results != {}:
            # power = np.exp(10*(self.results['avg']-0.25))
            # val_non_lin     = 1/(1+power)
            # val_non_lin_err = 10*power/((1+power)**2)*self.results['err']

            return [{'metric': 'avg_h_dist', 'dim': 'u', 
                     'val': self.results['avg'], 
                     'err': self.results['err'], 
                     'n_val': 1-self.results['avg'], 
                     'n_err': self.results['err'], 
                    #  'idx_val': val_non_lin, 
                    #  'idx_err': val_non_lin_err
                     }]
        else: pass
