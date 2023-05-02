# Description: Main script for hosting SD TabEval tool
# Author: Anton D. Lautrup
# Date: 16-11-2022

import numpy as np

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from .metrics import *
from .file_utils import empty_dict, create_results_file, add_to_results_file, convert_nummerical_pair, convert_nummerical_single

class SynthEval():
    def __init__(self, real ,hold_out=None, cat_cols=[], save_name = 'SynthEval_result_file', unique_thresh = 10, n_samples=None) -> None:
        self.real = real
        
        if hold_out is not None:
            self.hold_out = hold_out

            # Make sure columns and their order are the same.
            if len(real.columns) == len(hold_out.columns):
                hold_out = hold_out[real.columns.tolist()]
            assert real.columns.tolist() == hold_out.columns.tolist(), 'Columns in real and houldout dataframe are not the same'
        else:
            self.hold_out = None

        self.F1_type = 'micro' # Use {‘micro’, ‘macro’, ‘weighted’}

        self.categorical_columns = cat_cols
        self.numerical_columns = [column for column in real.columns if column not in cat_cols]

        self.save_flag = 0
        self.save_name = save_name + '.csv'

        self.fast_eval_flag = 0
        # Make sure the number of samples is equal in both datasets.
        # <ADL 23-11-2022 10:26> Not sure this is nessecary for our needs?
        # if n_samples is None:
        #     self.n_samples = min(len(self.real), len(self.fake))
        # elif len(fake) >= n_samples and len(real) >= n_samples:
        #     self.n_samples = n_samples
        # else:
        #     raise Exception(f'Make sure n_samples < len(fake/real). len(real): {len(real)}, len(fake): {len(fake)}')

        # self.real = self.real.sample(self.n_samples)
        # self.fake = self.fake.sample(self.n_samples)
        # assert len(self.real) == len(self.fake), f'len(real) != len(fake)'
        pass
    
    def _update_syn_data(self,fake):
        """Function for adding/updating the synthetic data"""
        self.fake = fake
        self.res_dict = empty_dict()

        if len(self.real.columns) == len(fake.columns):
            fake = fake[self.real.columns.tolist()]
        assert self.real.columns.tolist() == fake.columns.tolist(), 'Columns in real and fake dataframe are not the same'
        print('SynthEval: synthetic data read successfully')
        pass

    def save_results(self):
        if self.save_flag == 0:
            create_results_file(self.res_dict,self.save_name)
            self.save_flag = 1
        add_to_results_file(self.res_dict,self.save_name)
        pass

    def fast_eval(self,synthetic_dataframe, target):
        """This function is for running the quick version of TabEval, for model checkpoints etc."""
        self._update_syn_data(synthetic_dataframe)

        real, fake = convert_nummerical_pair(self.real,self.fake,self.categorical_columns)

        self._early_utility_analysis(real,fake,target)

        ks_dist, ks_p_val, ks_num_sig, ks_frac_sig = featurewise_ks_test(real, fake)
        self.ks_dist, self.ks_p_val, self.ks_num_sig, self.ks_frac_sig = ks_dist, ks_p_val, ks_num_sig, ks_frac_sig
        self.res_dict['Kolmogorov-Smirnov avg. dist'] = ks_dist
        self.res_dict['Kolmogorov-Smirnov avg. p-val'] = ks_p_val
        self.res_dict['Number of significant KS-tests at a=0.05'] = ks_num_sig
        self.res_dict['Fraction of significant KS-tests at a=0.05'] = ks_frac_sig

        H_dist = featurewise_hellinger_distance(real,fake,self.categorical_columns,self.numerical_columns)
        self.H_dist = H_dist
        self.res_dict['Average empirical Hellinger distance'] = H_dist

        mean_DCR = distance_to_closest_record(real,fake)
        self.mean_DCR = mean_DCR
        self.res_dict['Normed distance to closest record (DCR)'] = mean_DCR

        print('SynthEval: fast evaluation complete\n',
                "+-------------------------------+\n",
                "| avg. KS-dist         : %.4f |\n" % ks_dist,
                "| frac. of sig. tests  : %d      |\n"% ks_frac_sig,
                "| Avg. H-dist          : %.4f |\n" % H_dist,
                "| Normed DCR           : %.4f |\n" % mean_DCR,
                "+-------------------------------+\n"       
        )

        self.save_results()
        return [ks_dist,ks_frac_sig], H_dist, mean_DCR
    
    def full_eval(self, synthetic_data, target_col: str):
        """Main function for evaluate synthetic data"""
        ### Initialize the data
        self._update_syn_data(synthetic_data)
        real, fake = convert_nummerical_pair(self.real,self.fake,self.categorical_columns)
        
        if self.hold_out is not None: hout = convert_nummerical_single(self.hold_out,self.categorical_columns)
        else: hout = None

        # Initialize flags 
        EARL = QUAL = RESM = USEA = PRIV = False

        # Early utility analysis
        if len(self.numerical_columns) > 1:
            EARL = self._early_utility_analysis(real,fake,target_col)
        print("EARL ran sucessfully")
        # Quality analysis
        QUAL = self._quality_metrics(real,fake)
        print("QUAL ran sucessfully")
        # Resemblance analysis
        RESM = self._resemblance_metrics(real, fake)
        print("RESM ran sucessfully")
        # Usability analysis (ML-tests)
        USEA = self._usability_metrics(real, fake, target_col, hout)
        print("USEA ran sucessfully")
        # Primitive privacy
        PRIV = self._simple_privacy_mets(real, fake, hout)

        self.save_results()

        ### Print results to console
        self._print_results(EARL,QUAL,RESM,USEA,PRIV)
        pass
    
    def _early_utility_analysis(self,real,fake,target):
        """For sanity-checking the results"""
        dimensionwise_means(real,fake,self.numerical_columns,self.fast_eval_flag)
        principal_component_analysis(real,fake,self.numerical_columns,target,self.fast_eval_flag)

        self.fast_eval_flag += 1
        return True

    def _quality_metrics(self, real, fake):
        """Function for calculating pairwise statistics, correlations, Hellinger distance, confidence intervals etc."""

        corr_diff = correlation_matrix_difference(real,fake,self.numerical_columns)
        self.corr_diff = corr_diff
        self.res_dict['Correlation matrix differences (num only)'] = corr_diff

        mi_diff = mutual_information_matrix_difference(real,fake)
        self.mi_diff = mi_diff
        self.res_dict['Pairwise mutual information difference'] = mi_diff

        ks_dist, ks_p_val, ks_num_sig, ks_frac_sig = featurewise_ks_test(real, fake)
        self.ks_dist, self.ks_p_val, self.ks_num_sig, self.ks_frac_sig = ks_dist, ks_p_val, ks_num_sig, ks_frac_sig
        self.res_dict['Kolmogorov-Smirnov avg. dist'] = ks_dist
        self.res_dict['Kolmogorov-Smirnov avg. p-val'] = ks_p_val
        self.res_dict['Number of significant KS-tests at a=0.05'] = ks_num_sig
        self.res_dict['Fraction of significant KS-tests at a=0.05'] = ks_frac_sig
        return True
    
    def _resemblance_metrics(self, real, fake):
        """Function for calculating confidence interval overlaps, hellinger distance, pMSE and NNAA."""
        
        CIO, CIO_negatives = confidence_interval_overlap(real,fake,self.numerical_columns)
        self.CIO = CIO
        self.CIO_negatives = CIO_negatives
        self.res_dict['Average confidence interval overlap'] = CIO
        self.res_dict['Number of non overlapping COIs at 95pct'] = CIO_negatives

        H_dist = featurewise_hellinger_distance(real,fake,self.categorical_columns,self.numerical_columns)
        self.H_dist = H_dist
        self.res_dict['Average empirical Hellinger distance'] = H_dist

        pMSE, pMSE_acc = propensity_mean_square_error(real,fake)
        self.pMSE = pMSE
        self.pMSE_acc = pMSE_acc
        self.res_dict['Propensity Mean Squared Error (acc, pMSE)'] = np.array([pMSE_acc, pMSE])

        NNAA = adversarial_accuracy(real,fake)
        self.NNAA = NNAA
        self.res_dict['Nearest neighbour adversarial accuracy'] = NNAA
        return True

    def _usability_metrics(self, real, fake, target_col: str, hout=None):
        real_x, real_y = real.drop([target_col], axis=1), real[target_col]
        fake_x, fake_y = fake.drop([target_col], axis=1), fake[target_col]

        ### Build Classification the models
        svc=SVC(probability=True, kernel='linear')
        
        R_DT, F_DT = DecisionTreeClassifier(max_depth=15,random_state=42), DecisionTreeClassifier(max_depth=15,random_state=42)
        R_AB, F_AB = AdaBoostClassifier(n_estimators=10,learning_rate=1,random_state=42),AdaBoostClassifier(n_estimators=10,learning_rate=1,random_state=42)
        R_RF, F_RF = RandomForestClassifier(n_estimators=10,max_depth=15, random_state=42),RandomForestClassifier(n_estimators=10,max_depth=15, random_state=42)
        R_LG, F_LG = LogisticRegression(multi_class='auto', solver='saga', max_iter=50000, random_state=42), LogisticRegression(multi_class='auto', solver='saga', max_iter=50000, random_state=42)

        real_models = [R_DT, R_AB, R_RF, R_LG]
        fake_models = [F_DT, F_AB, F_RF, F_LG]

        ### Run 5-fold cross-validation
        kf = KFold(n_splits=5)
        res = np.zeros((len(real_models),2))
        for train_index, test_index in tqdm(kf.split(real_y),desc='USEA',total=5):
            real_x_train = real_x.iloc[train_index]
            real_x_test = real_x.iloc[test_index]
            real_y_train = real_y.iloc[train_index]
            real_y_test = real_y.iloc[test_index]
            fake_x_train = fake_x.iloc[train_index]
            fake_y_train = fake_y.iloc[train_index]

            res += class_test(real_models,fake_models,[real_x_train, real_y_train],
                                                            [fake_x_train, fake_y_train],
                                                            [real_x_test, real_y_test],self.F1_type)

        class_res = np.array(res)/5
        self.class_res = class_res
        self.res_dict['models trained on real data' ] = class_res[:,0]
        self.res_dict['models trained on fake data' ] = class_res[:,1]

        if hout is not None:
            hold_out_res = class_test(real_models, fake_models, [real_x, real_y],
                                                                            [fake_x, fake_y],
                                                                            [hout.drop([target_col],axis=1), hout[target_col]],self.F1_type)
            self.hold_out_res = hold_out_res
            self.res_dict['model trained on real data on holdout'] = hold_out_res[:,0]
            self.res_dict['model trained on fake data on holdout'] = hold_out_res[:,1]
                                   
        return True

    def _utility_score(self):
        """Function for calculating the overall utility score"""
        lst = []
        
        lst.append(1-np.tanh(self.corr_diff))
        lst.append(1-np.tanh(self.mi_diff))
        lst.append(1-self.ks_dist)
        lst.append(1-self.ks_frac_sig)
        lst.append(self.CIO)
        lst.append(1-self.H_dist)
        lst.append(1-self.pMSE/0.25)
        lst.append(1-self.NNAA)

        lst.append(1-np.mean(abs(self.class_res[:,0]-self.class_res[:,1])))
        lst.append(1-np.mean(abs(self.hold_out_res[:,0]-self.hold_out_res[:,1])))

        self.util_lst = lst
        self.res_dict['Overall utility score'] = np.mean(lst)
        return True

    def _simple_privacy_mets(self, real, fake, hout=None):
        """Calculate the mean distance to closest record (DCR) and hitting rate"""
        
        mean_DCR = distance_to_closest_record(real,fake)
        self.mean_DCR = mean_DCR
        self.res_dict['Normed distance to closest record (DCR)'] = mean_DCR

        hit_rate = hitting_rate(real,fake,self.categorical_columns)
        self.hit_rate = hit_rate
        self.res_dict['Hitting rate (thres = range(att)/30)'] = hit_rate

        ### Privacy loss (Nearest neighbour adversarial accuracy)
        if hout is not None:
            privloss = adversarial_accuracy(hout[real.columns],fake) -self.NNAA
        else: privloss = 0

        self.privloss = privloss
        self.res_dict['Privacy loss (discrepancy in NNAA)'] = privloss

        return True

    def _print_results(self, do_earl, do_qual, do_resm, do_usea, do_priv):

        print(
            "\nSynthEval Results \n",
                "=================================================================\n"
        )

        if do_qual:
            print(
                "Quality metrics:\n",
                "+---------------------------------------------------------------+\n",
                "| Correlation matrix difference (num only) :   %.4f           |\n" % (self.corr_diff),
                "| Pairwise mutual information difference   :   %.4f           |\n" % (self.mi_diff),
                "| Kolmogorov–Smirnov test                                       |\n",
                "|   -> avg. Kolmogorov–Smirnov distance    :   %.4f           |\n" % (self.ks_dist),
                "|   -> avg. Kolmogorov–Smirnov p-value     :   %.4f           |\n" % (self.ks_p_val),
                "|   -> # significant tests at a=0.05       :   %d                |\n" % (self.ks_num_sig),
                "|   -> fraction of significant tests       :   %.4f           |\n" % (self.ks_frac_sig),
                "+---------------------------------------------------------------+"
            )

        if do_resm:
            print(
                "Resemblance metrics:\n",
                "+---------------------------------------------------------------+\n",
                "| Average confidence interval overlap      :   %.4f           |\n" % (self.CIO),
                "|   -> # Non overlapping COIs at 95%%       :   %d                |\n" % (self.CIO_negatives),
                "| Average empirical Hellinger distance     :   %.4f           |\n" % (self.H_dist),
                "| Propensity Mean Squared Error (acc=%.2f) :   %.4f           |\n" % (self.pMSE_acc,self.pMSE),
                "| Nearest neighbour adversarial accuracy   :   %.4f           |\n" % (self.NNAA),
                "+---------------------------------------------------------------+"
            )

        if do_usea:
            res = self.class_res
            print(
                "Usability metrics:\n",
                "avg. of 5-fold cross val.:\n",
                "clasifier model              real_f1     fake_f1     |diff.|\n",
                "+---------------------------------------------------------------+\n",
                "| DecisionTreeClassifier :    %.4f      %.4f      %.4f    |\n" % (res[0,0], res[0,1], abs(res[0,0]-res[0,1])),
                "| AdaBoostClassifier     :    %.4f      %.4f      %.4f    |\n" % (res[1,0], res[1,1], abs(res[1,0]-res[1,1])),
                "| RandomForestClassifier :    %.4f      %.4f      %.4f    |\n" % (res[2,0], res[2,1], abs(res[2,0]-res[2,1])),
                "| LogisticRegression     :    %.4f      %.4f      %.4f    |\n" % (res[3,0], res[3,1], abs(res[3,0]-res[3,1])),
                "+---------------------------------------------------------------+\n",
                "| Average                :    %.4f      %.4f      %.4f    |\n" % (np.mean(res[:,0]),np.mean(res[:,1]),np.mean(abs(res[:,0]-res[:,1]))),
                "+---------------------------------------------------------------+"
            )

        if do_usea and (self.hold_out is not None):
            hres = self.hold_out_res
            print(
                " hold out data results:\n",
                "+---------------------------------------------------------------+\n",
                "| DecisionTreeClassifier :    %.4f      %.4f      %.4f    |\n" % (hres[0,0], hres[0,1], abs(hres[0,0]-hres[0,1])),
                "| AdaBoostClassifier     :    %.4f      %.4f      %.4f    |\n" % (hres[1,0], hres[1,1], abs(hres[1,0]-hres[1,1])),
                "| RandomForestClassifier :    %.4f      %.4f      %.4f    |\n" % (hres[2,0], hres[2,1], abs(hres[2,0]-hres[2,1])),
                "| LogisticRegression     :    %.4f      %.4f      %.4f    |\n" % (hres[3,0], hres[3,1], abs(hres[3,0]-hres[3,1])),
                "+---------------------------------------------------------------+\n",
                "| Average                :    %.4f      %.4f      %.4f    |\n" % (np.mean(hres[:,0]),np.mean(hres[:,1]),np.mean(abs(hres[:,0]-hres[:,1]))),
                "+---------------------------------------------------------------+"
            )

        if all([do_qual, do_resm, do_usea]):
            self._utility_score()
            print(
                " \n",
                "+---------------------------------------------------------------+\n",
                "| Overall utility score                    :   %.4f(%.4f)   |\n" % (np.mean(self.util_lst),np.std(self.util_lst,ddof=1)),
                "+---------------------------------------------------------------+"
            )
                
        if do_priv:
            print(
                "Privacy metrics:\n",
                "+---------------------------------------------------------------+\n",
                "| Normed distance to closest record (DCR)  :   %.4f           |\n" % (self.mean_DCR),
                "| Hitting rate (thres = range(att)/30)     :   %.4f           |\n" % (self.hit_rate),
                "| Privacy loss (discrepancy in NNAA)       :   %.4f           |\n" % (self.privloss),
                "+---------------------------------------------------------------+"
            )
        pass