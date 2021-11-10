# -*- coding: utf-8 -*-
from TonyModel import *


foldername = r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Peak_detection_data.xlsx"
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

input_folder = r'C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\Sin'
#input_folder = 

data_IO = extract_data(input_folder)
input_field = [x/10 for x in data_IO[0]]


#output data
output_ms_amp = []
output_ms_freq = []
output_v_amp = []

#initial peak parameters
ms = 1
v = 0
f = 7

#generate output field
for i in range(0, len(input_field)):
    input_field[i]
    if input_field[i] > max_field:
        input_field[i] = max_field
    if input_field[i] < min_field:
        input_field[i] = min_field
    if input_field[i] < vortex_remag_cutoff:
        new_peak = Vortex_field_new(ms, f, v, input_field[i], fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp)
        ms, f, v = new_peak
        output_ms_amp.append(new_peak[0])
        output_ms_freq.append(new_peak[1])
        output_v_amp.append(new_peak[2])
    elif input_field[i] >= vortex_remag_cutoff:
        new_peak = Remag_field_new(ms,v,input_field[i],input_field[i-1], fit_remag_ms, fit_remag_v)
        ms, v = new_peak
        output_ms_amp.append(new_peak[0])
        output_v_amp.append(new_peak[1])

output_ms_amp = normalise_data(output_ms_amp)
ms_data_output = normalise_data(data_IO[1])
v_data_output = normalise_data(data_IO[2])

mse_data = []
for i in range(0, len(output_ms_amp)):
    mse_data.append((output_ms_amp[i] - data_output[i])**2)

fig,ax = plt.subplots(4,2)
ax[0][0].plot(input_field)
ax[1][0].plot(output_ms_amp)
ax[2][0].plot(output_ms_amp)
ax[2][0].plot(ms_data_output)
ax[3][0].plot(mse_data)

ax[0][1].plot(input_field)
ax[1][1].plot(output_v_amp)
ax[2][1].plot(output_v_amp)
ax[2][1].plot(v_data_output)
ax[3][1].plot(mse_data)

plt.show()

