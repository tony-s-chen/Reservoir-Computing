# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 02:32:49 2021

@author: Tong
"""

from TonyModel import *


max_field = 23.5
min_field = 18
vortex_remag_cutoff = 19.5


exp_data = model_data()

input_folder = r'C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_glass'
#input_folder = 

data_IO = extract_data(input_folder)
input_field = [x/10 for x in data_IO[0]]


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

#generate output field
for i in range(0, len(input_field)):
    if input_field[i] > max_field:
        input_field[i] = max_field
    if input_field[i] < min_field:
        input_field[i] = min_field
    ms += model_step(exp_data, ms, input_field[i])
    print(ms)
    output_ms_amp.append(ms)
    
ms_data_output = normalise_data(data_IO[1])


fig,ax = plt.subplots(3,1)
ax[0].plot(input_field)

ax[1].plot(output_ms_amp)
ax[2].plot(ms_data_output)