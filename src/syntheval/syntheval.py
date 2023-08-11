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
from .file_utils import empty_dict, create_results_file, add_to_results_file, consistent_label_encoding, get_cat_variables

class SynthEval():
    def __init__(self, real ,hold_out=None, cat_cols=None, save_name = 'SynthEval_result_file', unique_thresh = 10) -> None:
        self.real = real

        if hold_out is not None:
            self.hold_out = hold_out

            # Make sure columns and their order are the same.
            if len(real.columns) == len(hold_out.columns):
                hold_out = hold_out[real.columns.tolist()]
            assert real.columns.tolist() == hold_out.columns.tolist(), 'Columns in real and houldout dataframe are not the same'
        else:
            self.hold_out = None

        # Infer categorical columns using unique_thresh
        if cat_cols is None:
            cat_cols = get_cat_variables(real, unique_thresh)
            print('SynthEval: inferred categorical columns...')
        
        self.categorical_columns = cat_cols
        self.numerical_columns = [column for column in real.columns if column not in cat_cols]

        ### Options
        self.mixed_correlation = True # Switch of for faster but less sensitive correlation matrix difference.
        self.permutation = True # Switch to False for faster but less sensitive KS test
        self.F1_type = 'micro' # Use {‘micro’, ‘macro’, ‘weighted’}
        self.knn_metric = 'gower' # Use {'gower', 'euclid'}

        self.save_flag = 0
        self.save_name = save_name + '.csv'

        self.fast_eval_flag = 0
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

    def fast_eval(self,synthetic_dataframe, target: str = None):
        """This function is for running the quick version of SynthEval, for model checkpoints etc."""
        self._update_syn_data(synthetic_dataframe)

        #real, fake = convert_nummerical_pair(self.real,self.fake,self.categorical_columns)
        CLE = consistent_label_encoding(self.real,self.fake,self.categorical_columns)
        real = CLE.encode(self.real)
        fake = CLE.encode(self.fake)

        self._early_utility_analysis(real,fake,target)

        if self.permutation == True:
            ks_dist, ks_p_val, ks_num_sig, ks_frac_sig = featurewise_ks_test(real, fake, cat_cols=self.categorical_columns)
        else:
            ks_dist, ks_p_val, ks_num_sig, ks_frac_sig = featurewise_ks_test(real, fake, cat_cols=[])
        self.ks_dist, self.ks_p_val, self.ks_num_sig, self.ks_frac_sig = ks_dist, ks_p_val, ks_num_sig, ks_frac_sig
        self.res_dict['Kolmogorov-Smirnov avg. dist'] = ks_dist
        self.res_dict['Kolmogorov-Smirnov avg. p-val'] = ks_p_val
        self.res_dict['Number of significant KS-tests at a=0.05'] = ks_num_sig
        self.res_dict['Fraction of significant KS-tests at a=0.05'] = ks_frac_sig

        H_dist = featurewise_hellinger_distance(real,fake,self.categorical_columns,self.numerical_columns)
        self.H_dist = H_dist
        self.res_dict['Average empirical Hellinger distance'] = H_dist

        DCR = distance_to_closest_record(real,fake,self.categorical_columns,self.numerical_columns,self.knn_metric)
        self.DCR = DCR
        self.res_dict['Normed distance to closest record (DCR)'] = DCR

        print('SynthEval: fast evaluation complete\n',
                "+------------------------------------------+\n",
                "| Avg. KS-dist         : %.4f SE(%.4f) |\n" % (ks_dist['avg'], ks_dist['err']),
                "| frac. of sig. tests  : %d                 |\n"% ks_frac_sig,
                "| Avg. H-dist          : %.4f SE(%.4f) |\n" % (H_dist['avg'], H_dist['err']),
                "| Normed DCR           : %.4f SE(%.4f) |\n" % (DCR['avg'], DCR['err']),
                "+------------------------------------------+\n"       
        )

        self.save_results()
        return ks_dist, ks_frac_sig, H_dist, DCR
    
    def priv_eval(self, synthetic_dataframe, target: str = None):
        """This function is for running only the privacy metrics, for model checkpoints etc."""
        self._update_syn_data(synthetic_dataframe)

        #real, fake = convert_nummerical_pair(self.real,self.fake,self.categorical_columns)
        CLE = consistent_label_encoding(self.real,self.fake,self.categorical_columns,self.hold_out)
        real = CLE.encode(self.real)
        fake = CLE.encode(self.fake)

        if self.hold_out is not None: 
            hout = CLE.encode(self.hold_out)
            NNAA = adversarial_accuracy(real, fake, self.categorical_columns, self.numerical_columns, self.knn_metric)
            self.NNAA = NNAA
            self.res_dict['Nearest neighbour adversarial accuracy'] = NNAA
        else: hout = None

        self._simple_privacy_mets(real, fake, hout)

        print('SynthEval: privacy evaluation complete\n',
                "+------------------------------------------+\n",
                "| Normed DCR           : %.4f SE(%.4f) |\n" % (self.DCR['avg'], self.DCR['err']),
                "| NN distance ratio    : %.4f SE(%.4f) |\n" % (self.NNDR['avg'],self.NNDR['err']),
                "| Hitting rate         : %.4f            |\n" % (self.hit_rate),
                "| epsilon identif.     : %.4f            |\n" % (self.eps_idf),
                "+------------------------------------------+"       
        )
        if self.hold_out is not None:
            print(
                "Privacy losses:\n",
                "+------------------------------------------+\n",
                "| NNAA discrepancy     : %.4f SE(%.4f) |\n" % (self.nnaa_loss['avg'], self.nnaa_loss['err']),
                "| NNDR discrepancy     : %.4f SE(%.4f) |\n" % (self.nndr_loss['avg'], self.nndr_loss['err']),
                "+------------------------------------------+",
            )
        
        self.save_results()
        return 

    def tuning_eval(self, synthetic_dataframe):
        self._update_syn_data(synthetic_dataframe)
        CLE = consistent_label_encoding(self.real,self.fake,self.categorical_columns,self.hold_out)
        real = CLE.encode(self.real)
        fake = CLE.encode(self.fake)

        H_dist = featurewise_hellinger_distance(real,fake,self.categorical_columns,self.numerical_columns)
        self.H_dist = H_dist
        self.res_dict['Average empirical Hellinger distance'] = H_dist

        eps_idf = epsilon_identifiability(real,fake,self.numerical_columns,self.categorical_columns,self.knn_metric)
        self.eps_idf = eps_idf
        self.res_dict['epsilon identifiability risk'] = eps_idf

        return H_dist['avg'], eps_idf

    def full_eval(self, synthetic_data, target_col: str = None):
        """Main function for evaluate synthetic data"""
        ### Initialize the data
        self._update_syn_data(synthetic_data)

        CLE = consistent_label_encoding(self.real,self.fake,self.categorical_columns)
        real = CLE.encode(self.real)
        fake = CLE.encode(self.fake)

        if self.hold_out is not None: hout = CLE.encode(self.hold_out)
        else: hout = None

        # Initialize flags 
        EARL = QUAL = RESM = USEA = PRIV = False

        # Early utility analysis
        if len(self.numerical_columns) > 1:
            EARL = self._early_utility_analysis(real,fake,target_col)
        print("EARL ran successfully")
        # Quality analysis
        QUAL = self._quality_metrics(real,fake)
        print("QUAL ran successfully")
        # Resemblance analysis
        RESM = self._resemblance_metrics(real, fake)
        print("RESM ran successfully")
        # Usability analysis (ML-tests)
        USEA = None if target_col is None else self._usability_metrics(real, fake, target_col, hout)
        print("USEA ran successfully")
        # Primitive privacy
        PRIV = self._simple_privacy_mets(real, fake, hout)

        self.save_results()

        ### Print results to console
        self._print_results(EARL,QUAL,RESM,USEA,PRIV)
        pass
    
    def _early_utility_analysis(self,real,fake,target):
        """For sanity-checking the results"""
        dimensionwise_means(real,fake,self.numerical_columns,self.fast_eval_flag)
        if target is not None:
            principal_component_analysis(real,fake,self.numerical_columns,target,self.fast_eval_flag)

        self.fast_eval_flag += 1
        return True

    def _quality_metrics(self, real, fake):
        """Function for calculating pairwise statistics, correlations, Hellinger distance, confidence intervals etc."""

        corr_diff = correlation_matrix_difference(real, fake, self.numerical_columns, self.categorical_columns, self.mixed_correlation)
        self.corr_diff = corr_diff
        self.res_dict['Correlation matrix differences (num only)'] = corr_diff

        mi_diff = mutual_information_matrix_difference(real,fake)
        self.mi_diff = mi_diff
        self.res_dict['Pairwise mutual information difference'] = mi_diff

        if self.permutation == True:
            ks_dist, ks_p_val, ks_num_sig, ks_frac_sig = featurewise_ks_test(real, fake,cat_cols=self.categorical_columns)
        else:
            ks_dist, ks_p_val, ks_num_sig, ks_frac_sig = featurewise_ks_test(real, fake,cat_cols=[])
        self.ks_dist, self.ks_p_val, self.ks_num_sig, self.ks_frac_sig = ks_dist, ks_p_val, ks_num_sig, ks_frac_sig
        self.res_dict['Kolmogorov-Smirnov avg. dist'] = ks_dist
        self.res_dict['Kolmogorov-Smirnov avg. p-val'] = ks_p_val
        self.res_dict['Number of significant KS-tests at a=0.05'] = ks_num_sig
        self.res_dict['Fraction of significant KS-tests at a=0.05'] = ks_frac_sig
        return True
    
    def _resemblance_metrics(self, real, fake):
        """Function for calculating confidence interval overlaps, hellinger distance, pMSE and NNAA."""
        
        CIO, num_CIO, frac_CIO = confidence_interval_overlap(real,fake,self.numerical_columns)
        self.CIO = CIO
        self.num_CIO = num_CIO
        self.frac_CIO = frac_CIO
        self.res_dict['Average confidence interval overlap'] = CIO
        self.res_dict['Number of non-overlapping CIs at 95pct'] = num_CIO
        self.res_dict['Fraction of non-overlapping CIs at 95pct'] = frac_CIO

        H_dist = featurewise_hellinger_distance(real,fake,self.categorical_columns,self.numerical_columns)
        self.H_dist = H_dist
        self.res_dict['Average empirical Hellinger distance'] = H_dist

        pMSE, pMSE_acc = propensity_mean_square_error(real,fake)
        self.pMSE = pMSE
        self.pMSE_acc = pMSE_acc
        self.res_dict['Propensity Mean Squared Error (pMSE)'] = pMSE
        self.res_dict['Propensity Mean Squared Error (acc)'] =  pMSE_acc

        NNAA = adversarial_accuracy(real, fake, self.categorical_columns, self.numerical_columns, self.knn_metric)
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
        res = []#np.zeros((len(real_models),2))
        for train_index, test_index in tqdm(kf.split(real_y),desc='USEA',total=5):
            real_x_train = real_x.iloc[train_index]
            real_x_test = real_x.iloc[test_index]
            real_y_train = real_y.iloc[train_index]
            real_y_test = real_y.iloc[test_index]
            fake_x_train = fake_x.iloc[train_index]
            fake_y_train = fake_y.iloc[train_index]

            res.append(class_test(real_models,fake_models,[real_x_train, real_y_train],
                                                          [fake_x_train, fake_y_train],
                                                          [real_x_test, real_y_test],self.F1_type))

        class_avg = np.mean(res,axis=0)
        class_err = np.std(res,axis=0,ddof=1)/np.sqrt(5)
        class_diff = np.abs(class_avg[0,:]-class_avg[1,:])
        class_diff_err = np.sqrt(class_err[0,:]**2+class_err[1,:]**2)

        self.class_res = class_avg
        self.class_diff = class_diff
        self.class_diff_err = class_diff_err
        self.res_dict['models trained on real data'] = {'avg':class_avg[0,:],'err':class_err[0,:]}
        self.res_dict['models trained on fake data'] = {'avg':class_avg[1,:],'err':class_err[1,:]}
        self.res_dict['f1 difference training data'] = {'avg':class_diff,'err':class_diff_err}

        if hout is not None:
            holdout_res = class_test(real_models, fake_models, [real_x, real_y],
                                                                [fake_x, fake_y],
                                                                [hout.drop([target_col],axis=1), hout[target_col]],self.F1_type)
            
            holdout_diff = np.abs(holdout_res[0,:]-holdout_res[1,:])

            self.holdout_res = holdout_res
            self.holdout_diff = holdout_diff
            self.res_dict['model trained on real data on holdout'] = holdout_res[0,:]
            self.res_dict['model trained on fake data on holdout'] = holdout_res[1,:]
            self.res_dict['f1 difference holdout data'] = holdout_diff
        else: self.holdout_diff=0
                                   
        return True

    def _utility_score(self):
        """Function for calculating the overall utility score"""
        lst = []
        
        lst.append(1-np.tanh(self.corr_diff))
        lst.append(1-np.tanh(self.mi_diff))
        lst.append(1-self.ks_dist['avg'])
        lst.append(1-self.ks_frac_sig)
        lst.append(self.CIO['avg'])
        lst.append(1-self.H_dist['avg'])
        lst.append(1-self.pMSE['avg']/0.25)
        lst.append(1-self.NNAA['avg'])

        lst.append(1-np.mean(self.class_diff))
        if self.hold_out is not None: lst.append(1-np.mean(self.holdout_diff))

        # prop_err = 1/10*np.sqrt(self.ks_dist['err']**2+self.CIO['err']**2+self.H_dist['err']**2+(self.pMSE['err']/0.25)**2+self.NNAA['err']**2+(0.25*np.sqrt(sum(self.class_diff_err**2)))**2+(np.std(self.holdout_diff,ddof=1)/np.sqrt(4))**2)
        # print(prop_err)

        res = {'avg':np.mean(lst),'err':np.std(lst,ddof=1)/np.sqrt(len(lst))}
        self.util_score = res
        self.res_dict['Overall utility score'] = res
        return True

    def _simple_privacy_mets(self, real, fake, hout=None):
        """Calculate the mean distance to closest record (DCR) and hitting rate"""
        
        DCR = distance_to_closest_record(real,fake,self.categorical_columns,self.numerical_columns, self.knn_metric)
        self.DCR = DCR
        self.res_dict['Normed distance to closest record (DCR)'] = DCR

        NNDR = nearest_neighbour_distance_ratio(real, fake, self.numerical_columns)
        self.NNDR = NNDR
        self.res_dict['Nearest neighbour distance ratio'] = NNDR

        hit_rate = hitting_rate(real,fake,self.categorical_columns)
        self.hit_rate = hit_rate
        self.res_dict['Hitting rate (thres = range(att)/30)'] = hit_rate

        eps_idf = epsilon_identifiability(real,fake,self.numerical_columns,self.categorical_columns,self.knn_metric)
        self.eps_idf = eps_idf
        self.res_dict['epsilon identifiability risk'] = eps_idf

        ### Privacy losses
        if hout is not None:
            NNAA_h = adversarial_accuracy(hout, fake, self.categorical_columns, self.numerical_columns, self.knn_metric)
            nnaa_loss_val = NNAA_h['avg'] - self.NNAA['avg']
            nnaa_loss_err = np.sqrt(self.NNAA['err']**2+NNAA_h['err']**2)
            nnaa_loss = {'avg': nnaa_loss_val, 'err': nnaa_loss_err}

            NNDR_h = nearest_neighbour_distance_ratio(hout, fake, self.numerical_columns)
            nndr_loss_val = NNDR['avg'] - NNDR_h['avg']
            nndr_loss_err = np.sqrt(NNDR['err']**2+NNDR_h['err']**2)
            nndr_loss = {'avg': nndr_loss_val, 'err': nndr_loss_err}

            self.nnaa_loss = nnaa_loss
            self.res_dict['Privacy loss (NNAA)'] = nnaa_loss
            self.nndr_loss = nndr_loss
            self.res_dict['Privacy loss (NNDR)'] = nndr_loss

        return True

    def _print_results(self, do_earl, do_qual, do_resm, do_usea, do_priv):

        print(
            "\nSynthEval Results \n",
                "=================================================================\n"
        )

        if do_qual:
            print(
                "Quality metrics:\n",
                "metric description                            value   error\n",                                   
                "+---------------------------------------------------------------+\n",
                "| Correlation matrix difference (num only) :   %.4f           |\n" % (self.corr_diff),
                "| Pairwise mutual information difference   :   %.4f           |\n" % (self.mi_diff),
                "| Kolmogorov–Smirnov test                                       |\n",
                "|   -> avg. Kolmogorov–Smirnov distance    :   %.4f  %.4f   |\n" % (self.ks_dist['avg'],self.ks_dist['err']),
                "|   -> avg. Kolmogorov–Smirnov p-value     :   %.4f  %.4f   |\n" % (self.ks_p_val['avg'],self.ks_p_val['err']),
                "|   -> # significant tests at a=0.05       :   %d                |\n" % (self.ks_num_sig),
                "|   -> fraction of significant tests       :   %.4f           |\n" % (self.ks_frac_sig),
                "+---------------------------------------------------------------+"
            )

        if do_resm:
            print(
                "Resemblance metrics:\n",
                "+---------------------------------------------------------------+\n",
                "| Average confidence interval overlap      :   %.4f  %.4f   |\n" % (self.CIO['avg'],self.CIO['err']),
                "|   -> # non-overlapping COIs at 95%%       :   %d                |\n" % (self.num_CIO),
                "|   -> fraction of non-overlapping CIs     :   %.4f           |\n" % (self.frac_CIO),                                
                "| Average empirical Hellinger distance     :   %.4f  %.4f   |\n" % (self.H_dist['avg'],self.H_dist['err']),
                "| Propensity Mean Squared Error (pMSE)     :   %.4f  %.4f   |\n" % (self.pMSE['avg'],self.pMSE['err']),
                "|   -> average pMSE classifier accuracy    :   %.4f  %.4f   |\n" % (self.pMSE_acc['avg'],self.pMSE_acc['err']),
                "| Nearest neighbour adversarial accuracy   :   %.4f  %.4f   |\n" % (self.NNAA['avg'],self.NNAA['err']),
                "+---------------------------------------------------------------+"
            )

        if do_usea:
            res = self.class_res
            print(
                "Usability metrics:\n",
                "avg. of 5-fold cross val.:\n",
                "clasifier model              acc_r   acc_f    |diff|  error\n",
                "+---------------------------------------------------------------+\n",
                "| DecisionTreeClassifier  :   %.4f  %.4f   %.4f  %.4f   |\n" % (res[0,0], res[1,0], self.class_diff[0], self.class_diff_err[0]),
                "| AdaBoostClassifier      :   %.4f  %.4f   %.4f  %.4f   |\n" % (res[0,1], res[1,1], self.class_diff[1], self.class_diff_err[1]),
                "| RandomForestClassifier  :   %.4f  %.4f   %.4f  %.4f   |\n" % (res[0,2], res[1,2], self.class_diff[2], self.class_diff_err[2]),
                "| LogisticRegression      :   %.4f  %.4f   %.4f  %.4f   |\n" % (res[0,3], res[1,3], self.class_diff[3], self.class_diff_err[3]),
                "+---------------------------------------------------------------+\n",
                "| Average                 :   %.4f  %.4f   %.4f  %.4f   |\n" % (np.mean(res[0,:]),np.mean(res[1,:]),np.mean(self.class_diff),0.25*np.sqrt(sum(self.class_diff_err**2))),
                "+---------------------------------------------------------------+"
            )

        if do_usea and (self.hold_out is not None):
            hres = self.holdout_res
            print(
                " hold out data results:\n",
                "+---------------------------------------------------------------+\n",
                "| DecisionTreeClassifier  :   %.4f  %.4f   %.4f           |\n" % (hres[0,0], hres[1,0], self.holdout_diff[0]),
                "| AdaBoostClassifier      :   %.4f  %.4f   %.4f           |\n" % (hres[0,1], hres[1,1], self.holdout_diff[1]),
                "| RandomForestClassifier  :   %.4f  %.4f   %.4f           |\n" % (hres[0,2], hres[1,2], self.holdout_diff[2]),
                "| LogisticRegression      :   %.4f  %.4f   %.4f           |\n" % (hres[0,3], hres[1,3], self.holdout_diff[3]),
                "+---------------------------------------------------------------+\n",
                "| Average                 :   %.4f  %.4f   %.4f  %.4f   |\n" % (np.mean(hres[:,0]),np.mean(hres[:,1]),np.mean(self.holdout_diff),np.std(self.holdout_diff,ddof=1)/np.sqrt(4)),
                "+---------------------------------------------------------------+"
            )

        if all([do_qual, do_resm, do_usea]):
            self._utility_score()
            print(
                "                                                        SE\n",
                "+---------------------------------------------------------------+\n",
                "| Overall utility score                    :   %.4f  %.4f   |\n" % (self.util_score['avg'],self.util_score['err']),
                "+---------------------------------------------------------------+"
            )
                
        if do_priv:
            print(
                "Privacy metrics:\n",
                "+---------------------------------------------------------------+\n",
                "| Normed distance to closest record (DCR)  :   %.4f  %.4f   |\n" % (self.DCR['avg'], self.DCR['err']),
                "| Nearest neighbour distance ratio (NNDR)  :   %.4f  %.4f   |\n" % (self.NNDR['avg'], self.NNDR['err']),
                "| Hitting rate (thres = range(att)/30)     :   %.4f           |\n" % (self.hit_rate),
                "| Epsilon identifiability risk             :   %.4f           |\n" % (self.eps_idf),
                "+---------------------------------------------------------------+"
            )
        if do_priv and (self.hold_out is not None):
            print(
                " | Privacy loss (NNAA)                      :   %.4f  %.4f   |\n" % (self.nnaa_loss['avg'],self.nnaa_loss['err']),
                "| Privacy loss (NNDR)                      :   %.4f  %.4f   |\n" % (self.nndr_loss['avg'],self.nndr_loss['err']),
                "+---------------------------------------------------------------+"
            )
        pass