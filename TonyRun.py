# -*- coding: utf-8 -*-
from TonyModel import *


foldername = r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Peak_detection_data.xlsx"
r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx"
#foldername = 

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


#input_field = []
#n=1
#m=1
#period = 30
#for i in range(1000):
#	input_field.append((np.sin(i*2*3.1415*n/(period))**m)*0.5*(max_field-min_field)+0.5*(min_field+max_field))
    
#input data

input_folder = r'C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin'

input_folder = r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw"

data_IO = extract_data(input_folder)
input_field = [x/10 for x in data_IO[0]]
ms_data_output = normalise_data(data_IO[1])
v_data_output = normalise_data(data_IO[2])

run_model(input_field,max_field,min_field,vortex_remag_cutoff,fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp,fit_remag_ms,fit_remag_v,ms_data_output)
#output data
output_ms_amp = []
output_ms_freq = []
output_v_amp = []
output_ms_weighted = []

#initial peak parameters
ms = 1
ms_test = 1
v = 0
f = 7

'''
coeff calc
'''
#a_range = np.linspace(0,8,401)
#a_error = []
#for a in a_range:
#    output_ms_weighted = []
#    
#    #initial peak parameters
#    ms = 1
#    ms_test = 1
#    error_sum = 0
#    
#    for i in range(0, len(input_field)):
#        if input_field[i] > max_field:
#            input_field[i] = max_field
#        if input_field[i] < min_field:
#            input_field[i] = min_field
#        vortex_diff = Vortex_field_new(ms_test, f, v, input_field[i], fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp)[-1]
#        remag_diff = Remag_field_new(ms_test,v,input_field[i],input_field[i-1], fit_remag_ms, fit_remag_v)[-1]
#        vortex_weight, remag_weight = vortex_remag_weight(input_field[i],vortex_remag_cutoff,min_field, max_field,a)
#        ms_test += vortex_weight*vortex_diff + remag_weight*remag_diff
#        output_ms_weighted.append(ms_test)
#        error_sum += (output_ms_weighted[i] - ms_data_output[i])**2
#    print(a)
#    a_error.append(error_sum)
#print(a_error)
#a_optimum = a_range[np.argmin(a_error)]
#print(a_optimum)
#plt.figure()
#plt.plot(a_range,a_error)
#plt.show()

#a_optimum = 2
##output data
#output_ms_amp = []
#output_ms_freq = []
#output_v_amp = []
#output_ms_weighted = []
#
##initial peak parameters
#ms = 1
#ms_test = 1
#v = 0
#f = 7
#
##generate output field
#for i in range(0, len(input_field)):
#    input_field[i]
#    if input_field[i] > max_field:
#        input_field[i] = max_field
#    if input_field[i] < min_field:
#        input_field[i] = min_field
#    if input_field[i] < vortex_remag_cutoff:
#        new_peak = Vortex_field_new(ms, f, v, input_field[i], fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp)
#        ms, f, v = new_peak[0:3]
#        output_ms_amp.append(new_peak[0])
#        output_ms_freq.append(new_peak[1])
#        output_v_amp.append(new_peak[2])
#    elif input_field[i] >= vortex_remag_cutoff:
#        new_peak = Remag_field_new(ms,v,input_field[i],input_field[i-1], fit_remag_ms, fit_remag_v)
#        ms, v = new_peak[0:2]
#        output_ms_amp.append(new_peak[0])
#        output_v_amp.append(new_peak[1])
#    vortex_diff = Vortex_field_new(ms_test, f, v, input_field[i], fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp)[-1]
#    remag_diff = Remag_field_new(ms_test,v,input_field[i],input_field[i-1], fit_remag_ms, fit_remag_v)[-1]
#    vortex_weight, remag_weight = vortex_remag_weight(input_field[i],vortex_remag_cutoff,min_field, max_field,a_optimum)
#    #print(vortex_weight, remag_weight)
#    ms_test += vortex_weight*vortex_diff + remag_weight*remag_diff
#    output_ms_weighted.append(ms_test)
#    
#    
#
#output_ms_amp = normalise_data(output_ms_amp)
#
#mse_sum = 0
#for i in range(0, len(output_ms_weighted)):
#    mse_sum += (output_ms_weighted[i] - ms_data_output[i])**2
#    
#print(mse_sum)
#    
#output_ms_weighted = normalise_data(output_ms_weighted)
#
#mse_data = []
#for i in range(0, len(output_ms_amp)):
#    mse_data.append((output_ms_amp[i] - ms_data_output[i])**2)
#
#mse_weighted_data = []
#for i in range(0, len(output_ms_weighted)):
#    mse_weighted_data.append((output_ms_weighted[i] - ms_data_output[i])**2)
#    
#
#
#fig,ax = plt.subplots(5,1)
#ax[0].plot(input_field)
#ax[0].set_title("Input field")
#
#ax[1].plot(ms_data_output)
#ax[1].set_title("Experimental Output")
#
#ax[2].plot(ms_data_output, label = "Experimental")
#ax[2].plot(output_ms_amp, label = "Computational")
#ax[2].set_title("Computational Output")
#ax[2].legend()
#
#ax[3].plot(ms_data_output, label = "Experimental")
#ax[3].plot(output_ms_weighted, label = "Computational")
#ax[3].set_title("Weighted Computational Output")
#ax[3].legend()
#
#ax[4].plot(mse_data, label = "Not weighted")
#ax[4].plot(mse_weighted_data, label = "weighted")
#ax[4].set_title("Error Squared")
#ax[4].legend()
#
#plt.show()

#max_peak = extract_peak(input_folder)
#max_peak = normalise_data(max_peak)

#fig,ax = plt.subplots(3,1)
#ax[0].plot(ms_data_output)
#ax[0].plot(output_ms_amp)
#ax[1].plot(max_peak)
#ax[1].plot(output_ms_amp)
#ax[2].plot(ms_data_output)
#ax[2].plot(max_peak)