# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:26:19 2021

@author: Tong
"""
from TonyModel import *
from TonyRegressionModule import *

foldername = r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Peak_detection_data.xlsx"

#calculating the fits
fit_training_ms_amp = fit_training_data(foldername)
fit_training_ms_freq = fit_training_frequency_data(foldername)
fit_training_v_amp = fit_training_data_vortex(foldername)

fit_remag_ms = fit_remag_data(foldername)
fit_remag_v = fit_remag_data_vortex(foldername)

#input field data
max_field = 23.5
min_field = 18
vortex_remag_cutoff = 19.5#(max_field + min_field)/2

input_folder = r'C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin'
input_folder2 = r'C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass'
input_folder3 = r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw"
input_folder4 = r"C:\Users\Tong\Downloads\All_data\HDS_GS\\Random_exp"
#input_folder = r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw"

data_IO = extract_data(input_folder, 'IQ210')
input_field = [x/10 for x in data_IO[0]]
data_IO2 = extract_data(input_folder2, 'IQ210')
input_field2 = [x/10 for x in data_IO2[0]]
data_IO3 = extract_data(input_folder3, 'IQ217')
input_field3 = [x/10 for x in data_IO3[0]]
input_field  = input_field + input_field2 + input_field3
ms_data_output = data_IO[1] + data_IO2[1] +data_IO3[1]

data_IO4 = extract_data(input_folder4, 'IQ209')
input_field4 = [x/10 for x in data_IO4[0]]
#v_data_output = normalise_data(data_IO[2])

gen_output_train = run_model(input_field,max_field,min_field,vortex_remag_cutoff,fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp,fit_remag_ms,fit_remag_v,ms_data_output)

gen_output_test = run_model(input_field4,max_field,min_field,vortex_remag_cutoff,fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp,fit_remag_ms,fit_remag_v,data_IO4[1])
fig,ax = plt.subplots(2,1)
ax[0].plot(gen_output_test[0])
ax[1].plot(ms_data_output)
train_data = df_data(gen_output_train[0],10,10,gen_output_train[1], ms_data_output)
test_data = df_data(gen_output_test[0],10,10,gen_output_test[1],data_IO4[1])
print(test_data)
print(train_data)
reg = regression(train_data[0], train_data[1], test_data[0], test_data[1], "Lasso", show_plot = True)
#
#fig,ax = plt.subplots(2,1)
#
#ax[0].plot(test_data[0]['Current_input'])
#ax[0].set_title('Input ECG')
#
#
### Plot the test data
#ax[1].plot(reg[2])
#ax[1].set_title('Output')
#plt.show()