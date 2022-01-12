# -*- coding: utf-8 -*-
"""
@author: Tony
"""

from TonyRegressionModule import *
from sklearn.model_selection import train_test_split

#%% train data on all three sin/MG/saw data an prediction on random_exp
train_data = data_extractor_MG(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass\all_data.xlsx", 'IQ210', 10, 10)
train_data1, train_data2 = train_test_split(train_data[0])
regression(train_data1, train_data[1], train_data2, train_data[1], "Lasso", show_plot = True)
