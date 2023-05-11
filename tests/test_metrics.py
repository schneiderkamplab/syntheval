import unittest

import math
import pandas as pd

from syntheval.metrics import *
from syntheval.file_utils import convert_nummerical_single,convert_nummerical_pair

load_dir = 'guides/example/'
filename = 'penguins'

df_real = pd.read_csv(load_dir + filename + '_train.csv')
df_fake = pd.read_csv(load_dir + filename + '_BN_syn.csv').round(1)
df_test = pd.read_csv(load_dir + filename + '_test.csv')

cat_cols = ['species','island','sex']
num_cols = [col for col in df_real.columns if col not in cat_cols]

data1 = pd.DataFrame([[1,1,'cls1'],[2,2,'cls2'],[3,2,'cls2'],[4,2,'cls2']],columns=['var1','var2','cls'])
data2 = pd.DataFrame([[1,2,'cls1'],[2,3,'cls1'],[3,1,'cls2'],[3,2,'cls2']],columns=['var1','var2','cls'])

class TestCorr(unittest.TestCase):
    def test_same(self):
        self.assertEqual(correlation_matrix_difference(data1, data1,['var1', 'var2']), 0.0)

    def test_sym(self):
        self.assertEqual(correlation_matrix_difference(data1, data2,['var1', 'var2']), correlation_matrix_difference(data2, data1, ['var1', 'var2']))

    def test_wrongdata(self):
        self.assertRaises(KeyError, correlation_matrix_difference, data1, df_real, ['var1', 'var2'])
    
    def test_wrongtype(self):
        self.assertEqual(correlation_matrix_difference(data1, data2,['var1', 'cls']), 0.0)

class TestMI(unittest.TestCase):
    def test_same(self):
        self.assertEqual(mutual_information_matrix_difference(data1, data1), 0.0)

    def test_sym(self):
        self.assertEqual(mutual_information_matrix_difference(data1, data2), mutual_information_matrix_difference(data2, data1))

    def test_wrongdata(self):
        self.assertTrue(math.isnan(mutual_information_matrix_difference(data1,df_real)))

class TestKS(unittest.TestCase):
    def test_same(self):
        ks_dist, ks_p_val, ks_num_sig, ks_frac_sig = featurewise_ks_test(data1,data1)
        self.assertEqual(ks_dist,0.0)
        self.assertEqual(ks_p_val,1.0)
        self.assertEqual(ks_num_sig,0)
        self.assertEqual(ks_frac_sig,0.0)
    
    def test_sym(self):
        self.assertEqual(featurewise_ks_test(data1, data2), featurewise_ks_test(data2, data1))
    
    def test_wrongdata(self):
        self.assertRaises(KeyError, featurewise_ks_test, data1, df_real)

class TestHellinger(unittest.TestCase):
    def test_num_cats(self):
        self.assertRaises(TypeError, featurewise_hellinger_distance, data1, data1,['cls'],['var1','var2'])

    def test_same(self):
        df_temp = convert_nummerical_single(df_real,cat_cols)
        self.assertEqual(featurewise_hellinger_distance(df_temp,df_temp,cat_cols,num_cols),0.0)

class TestCIO(unittest.TestCase):
    def test_num_cats(self):
        self.assertRaises(TypeError, confidence_interval_overlap, data1, data1, data1.columns)

    def test_same(self):
        df_temp = convert_nummerical_single(df_real,cat_cols)
        self.assertEqual(confidence_interval_overlap(df_temp,df_temp,num_cols),(1.0, 0))

class TestpMSE(unittest.TestCase):
    def test_num_cats(self):
        self.assertRaises(ValueError, propensity_mean_square_error, df_real,df_fake)

    def test_pMSE(self):
        df_real_tmp, df_fake_tmp = convert_nummerical_pair(df_real,df_fake,cat_cols)
        same = propensity_mean_square_error(df_real_tmp, df_real_tmp)[0]

        self.assertLess(same,0.20)
        self.assertGreater(propensity_mean_square_error(df_real_tmp, df_fake_tmp)[0],same)

class TestNNAA(unittest.TestCase):
    def test_NNAA(self):
        df_real_tmp, df_fake_tmp = convert_nummerical_pair(df_real,df_fake,cat_cols)
        self.assertLess(adversarial_accuracy(df_real_tmp,df_fake_tmp), 0.5)

    def test_NNAA_repeats(self):
        df_real_tmp, df_fake_tmp = convert_nummerical_pair(df_test,df_fake,cat_cols)
        self.assertGreater(adversarial_accuracy(df_real_tmp,df_fake_tmp),0.5)

class TestDCR(unittest.TestCase):
    def test_same(self):
        df_tmp = convert_nummerical_single(df_real, cat_cols)
        self.assertEqual(distance_to_closest_record(df_tmp, df_tmp), 1.0)

    def test_DCR(self):
        df_real_tmp, df_fake_tmp = convert_nummerical_pair(data1,data2,['cls'])
        distance_to_closest_record(df_real_tmp, df_fake_tmp)

class TestHR(unittest.TestCase):
    def test_same(self):
        df_tmp = convert_nummerical_single(df_real, cat_cols)
        self.assertEqual(hitting_rate(df_tmp, df_tmp, cat_cols), 1.0)

    def test_HR(self):
        df_real_tmp, df_fake_tmp = convert_nummerical_pair(data1,data2,['cls'])
        self.assertEqual(hitting_rate(df_real_tmp, df_fake_tmp, 'cat'),0.0)

if __name__ == '__main__':
    unittest.main()