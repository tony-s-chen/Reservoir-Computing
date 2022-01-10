# -*- coding: utf-8 -*-
"""
@author: Tony
"""

from TonyRegressionModule import *

#%% train data on all three sin/MG/saw data an prediction on random_exp
train_data_1 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin\all_data.xlsx", 'IQ210', 10, 10)
train_data_2 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass\all_data.xlsx", 'IQ210', 10, 10)
train_data_3 = data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx", 'IQ217', 10, 10)
train_data = train_data_1[0].append(train_data_2[0])
train_data = train_data.append(train_data_3[0])
test_data= data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\\Random_exp\all_data.xlsx", 'IQ209', 10, 10)
regression(train_data, train_data_1[1], test_data[0], test_data[1], "Lasso", show_plot = True)

#%% grid search for the combintation of prev_input and prev_output number that provides the lowest MSE - takes long time to run
    
prev_values = [0,2]
MSE_test_array = []
MSE_train_array = []
values_array = []

for prev_inputs in prev_values:
    for prev_outputs in prev_values:#
        
        train_data_1 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin\all_data.xlsx", 'IQ210', prev_inputs, prev_outputs)
        train_data_2 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass\all_data.xlsx", 'IQ210', prev_inputs, prev_outputs)
        train_data_3 = data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx", 'IQ217', prev_inputs, prev_outputs)
        train_data = train_data_1[0].append(train_data_2[0])
        train_data = train_data.append(train_data_3[0])
        test_data= data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\\Random_exp\all_data.xlsx", 'IQ209', prev_inputs, prev_outputs)
        reg_model = regression(train_data, train_data_1[1], test_data[0], test_data[1], "Lasso")
        MSE_train_array.append(reg_model[3])
        MSE_test_array.append(reg_model[4])
        values_array.append([prev_inputs,prev_outputs])
    
print(MSE_train_array)
print(MSE_test_array)
print(values_array)
print(values_array[np.argmin(MSE_test_array)])
    
#%% plotting the extracted minimum point and testing it - this was tested using an empty dataframe to see how welll the model performs when no experimental output is provided


prev_inputs = 2
prev_outputs = 2
train_data_1 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin\all_data.xlsx", 'IQ210', prev_inputs, prev_outputs)
train_data_2 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass\all_data.xlsx", 'IQ210', prev_inputs, prev_outputs)
train_data_3 = data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx", 'IQ217', prev_inputs, prev_outputs)
train_data = train_data_1[0].append(train_data_2[0])
train_data = train_data.append(train_data_3[0])
test_data = empty_df(r"C:\Users\Tong\Downloads\All_data\HDS_GS\\Random_exp\all_data.xlsx", 'IQ209', prev_inputs, prev_outputs)
reg_model = regression2(train_data, train_data_1[1], test_data[0], test_data[1],prev_inputs, prev_outputs, "Lasso", show_plot = True)

#%% comparing the weighting matrix for different prev_step lengths.

#prev_steps = [3,5, 8, 10, 12, 15, 18, 20]
#fig,ax = plt.subplots(3,1)
#
#
#
### Assign the target point to be the frequency bin
#for prev_length in prev_steps:
#    train_data_1 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin\all_data.xlsx", 'IQ210', prev_length, prev_length)
#    train_data_2 = data_extractor(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass\all_data.xlsx", 'IQ210', prev_length, prev_length) 
#    train_data_3 = data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx", 'IQ217', prev_length, prev_length)
#    train_data = train_data_1[0].append(train_data_2[0])
#    train_data = train_data.append(train_data_3[0])
#    test_data= data_extractor(r"C:\Users\Tong\Downloads\All_data\HDS_GS\\Random_exp\all_data.xlsx", 'IQ209', prev_length, prev_length)
#    reg = regression(train_data, train_data_1[1], test_data[0], test_data[1], "Lasso")[0]
#
#    ax[0].plot(prev_length, reg.coef_[0], 'x', label = f"{prev_length} values")
#    ax[0].set_xlabel("Current input weighting")
#    ax[1].plot(reg.coef_[1:prev_length+1], label = f"{prev_length} values")
#    ax[1].set_ylabel("Weighting")
#    ax[2].plot(reg.coef_[prev_length+1:], label = f"{prev_length} values")
#    ax[2].set_xlabel("Time steps ago")
#    ax[2].set_ylabel("Weighting")
#    print(prev_length, reg.coef_)
#    print(reg.coef_[0])
#    print(reg.coef_[1:prev_length+1])
#    print(reg.coef_[prev_length+1:])
#    
#    
#plt.legend()
#plt.show()