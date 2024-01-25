# Description: Script for hideing all the ugly console printing stuff from the main SynthEval class 
# Author: Anton D. Lautrup
# Date: 25-08-2023

import numpy as np

def print_results_to_console(utility_output_txt,privacy_output_txt):#,scores):
    print("""\

SynthEval results
=================================================================
""")

    if utility_output_txt != '':
        print("""\
Utility metric description                    value   error                                 
+---------------------------------------------------------------+"""
        )
        print(utility_output_txt.rstrip())
        print("""\
+---------------------------------------------------------------+
    """)

    if privacy_output_txt != '':
        print("""\
Privacy metric description                    value   error                                 
+---------------------------------------------------------------+"""
            )
        print(privacy_output_txt.rstrip())
        print("""\
+---------------------------------------------------------------+
    """)
#     print("""\
# +---------------------------------------------------------------+"""
#         )
#     if not scores['utility']['val'] == []:
#         scores_lst = np.sqrt(sum(np.square(scores['utility']['err'])))/len(scores['utility']['val'])
#         print("""\
# | Utility index (avg. of %2d scores)        :   %.4f  %.4f   |""" % (len(scores['utility']['val']),np.mean(scores['utility']['val']), scores_lst)
#         )
        
#     if not scores['privacy']['val'] == []:
#         scores_lst = np.sqrt(sum(np.square(scores['privacy']['err'])))/len(scores['privacy']['val'])
#         print("""\
# | Privacy index (avg. of %2d scores)        :   %.4f  %.4f   |""" % (len(scores['privacy']['val']),np.mean(scores['privacy']['val']), scores_lst)
#         )

#     print("""\
# +---------------------------------------------------------------+"""
#         )