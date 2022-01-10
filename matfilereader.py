# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:16:09 2021

@author: Tong
"""
from TonyRegressionModule import *


train_data_1 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin\all_data.xlsx", 'IQ210', 2, 2)
train_data_2 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass\all_data.xlsx", 'IQ210', 2, 2)
train_data_3 = data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx", 'IQ217', 2, 2)
train_data = train_data_1[0].append(train_data_2[0])
train_data = train_data.append(train_data_3[0])

#test_data = mat_extract(r'C:\Users\Tong\Downloads\training2017\training2017\A00020.mat',10,10)
test_data = mat_extract(r'C:\Users\Tong\Downloads\archive\Training_2\Q0001.mat',2 ,2)
#test_data = empty_df(r"C:\Users\Tong\Downloads\All_data\HDS_GS\\Random_exp\all_data.xlsx", 'IQ209', 2, 2)
reg = regression2(train_data, train_data_1[1], test_data[0], test_data[1],2,2, "Lasso", show_plot = True)
#%%
fig,ax = plt.subplots(2,1)

ax[0].plot(test_data[0]['Current_input'])
ax[0].set_title('Input ECG')


## Plot the test data
ax[1].plot(test_data[0]['Current_input']/max(test_data[0]['Current_input'])*np.abs(max(reg[2])), label = "Scaled Input")
ax[1].plot(reg[2], label = "Output")
ax[1].set_title('Comparison')
ax[1].legend()
plt.show()