# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 01:38:11 2021

@author: Tong
"""

import _pickle as pickle
import numpy as np
import scipy as sp
import matplotlib
import os
#import pylab
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#import tkinter
#from scipy.signal import find_peaks
#import csv
import pandas as pd    
import csv
#import operator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import imageio
#import cv2
#import shutil
#import plotly.graph_objects as go

#import openpyxl
#from scipy import signal
#from openpyxl.styles import Color, PatternFill, Font, Border
#from openpyxl.styles import colors
#from openpyxl.cell import Cell
#import matplotlib.pyplot as plt
#import matplotlib.tri as mtri
#import numpy as np
#from matplotlib import cm
#import seaborn as sns
#from matplotlib.mlab import griddata
#from scipy.stats import gaussian_kde
#import matplotlib.ticker as ticker
#from scipy.optimize import curve_fit
#from scipy.signal import square, sawtooth, correlate
#import matplotlib.ticker as ticker
#from scipy.interpolate import interp1d
#from PIL import Image, ImageOps
#import xlsxwriter
#import math
#import gzip
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split # for the initial split to a train set and a untouched test set 
import random
from sklearn.model_selection import TimeSeriesSplit # for roll forward cross vallidation
import collections
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import math
from openpyxl import load_workbook
import sys
from sklearn.metrics import mean_squared_error
#prev_inputs = 10 ## Number of previous inputs to use for the ridge regression
#prev_outputs = 10 ## Number of previous outputs to use for the rideg regression
target_bin2 = 'IQ210'
target_bin = 'IQ210'

#data = train, data2 = test
#data = r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx"
data2 = r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Sin\all_data.xlsx"
data = r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Random\all_data.xlsx"
data3 = r"C:\Users\Tong\Downloads\All_data\HDS_GS\Inv_Saw\all_data.xlsx"

prev_values = [3,5,8,10]
MSE_test_array = []
MSE_train_array = []

for prev_inputs in prev_values:
    for prev_outputs in prev_values:
    
        df=pd.read_excel(data,sheet_name = 'LP_Scale0')
        df2=pd.read_excel(data2,sheet_name = 'LP_Scale0')
        
        df_data = pd.DataFrame()
        df_data2 = pd.DataFrame()
        
        ## Assign the target point to be the frequency bin
        
        df_data['Target'] = df[target_bin]#.shift(max([prev_outputs,prev_inputs]))
        df_data['Current_input'] = df['H_app2']
        
        df_data2['Target'] = df2[target_bin2]#.shift(max([prev_outputs,prev_inputs]))
        df_data2['Current_input'] = df2['H_app2']
        
        reservoir_output_names = []
        
        ## Create empty df columns for input and outputs
        for i in range(prev_outputs):
            df_data['Output'+str(i)] = 0
            df_data2['Output'+str(i)] = 0
            reservoir_output_names.append('Output'+str(i))
        
        for i in range(prev_inputs):
            df_data['Input'+str(i)] = 0
            df_data2['Input'+str(i)] = 0
            reservoir_output_names.append('Input'+str(i))
        
        ## Include the current input as a parameter
        reservoir_output_names.append('Current_input')
        
        ## Add in the previous input and output values 
        ## At the moment it is configured to include the only the previous outputs from the target bin, you can add different bins with other for loops - I can help with this
        for i in range(len(df_data)):
            for j in range(prev_inputs):
                df_data['Input'+str(j)].iloc[i] = df['H_app2'].iloc[i-j-1]
            for j in range(prev_outputs):
                df_data['Output'+str(j)].iloc[i] = df[target_bin].iloc[i-j-1]
        
        
        for i in range(len(df_data2)):
            for j in range(prev_inputs):
                df_data2['Input'+str(j)].iloc[i] = df2['H_app2'].iloc[i-j-1]
            for j in range(prev_outputs):
                df_data2['Output'+str(j)].iloc[i] = df2[target_bin2].iloc[i-j-1]
                
        ## Remove nans
        df_data = df_data.tail(len(df_data)-max([prev_outputs,prev_inputs]))
        df_data2 = df_data2.tail(len(df_data2)-max([prev_outputs,prev_inputs]))

        df_scale = 1
        df_scale2 =1
        df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
        df_data2["Target"] = df_data2["Target"].div(df_data2["Target"].max())

        df_data = df_data.append(df_data2)
        
        ## Here you are splitting your data into 'train' and 'test' bins so that you can work out the model and test on previously unseen data.
        ## x_train - all of the inputs and outputs which you will use in the model
        ## y_train - the target values
        
        
        testsize = 0.7 ## Fraction of your data for testing
        x_train, x_test, y_train, y_test=train_test_split(df_data[reservoir_output_names], df_data['Target'], test_size=testsize,shuffle=False)
        x_train2, x_test2, y_train2, y_test2=train_test_split(df_data2[reservoir_output_names], df_data2['Target'], test_size=testsize,shuffle=False)
        #
        #print('x_train',x_train)
        #print('x_test',x_test)
        #print('y_train',y_train)
        #print('y_test',y_test)
        # 
        
        ## Create the ridge regression model - there are other models available on the sklearn package. You can play around with these
        b = 1e-4
        model=Lasso(alpha = b, fit_intercept = True,copy_X = True)
        
        ## Fit the model
        reg = model.fit(df_data[reservoir_output_names], df_data['Target'])
        
        ## Print the weights and intercept
#        print("ridge coeffs")
#        print(reg.coef_)
#        print(reg.intercept_)
        
        ## Test on the test data
        output_test = reg.predict(df_data2[reservoir_output_names])
        output_train = reg.predict(df_data[reservoir_output_names])
        
        
        MSE_train = mean_squared_error(df_data['Target'],output_train)
        MSE_test = mean_squared_error(df_data2['Target'],output_test)
    
        MSE_train_array.append(MSE_train)
        MSE_test_array.append(MSE_test)

print(MSE_train_array)
print(MSE_test_array)
## Check that the MSE is better than just copying the previous step
#MSE_prev_step = mean_squared_error(y_test,x_test['Output0'])
## Plot the results
#fig,ax = plt.subplots(3,1)

## PLot the train data
#ax[0].plot(np.linspace(0,len(output_train),len(output_train)), df_data['Target'],c='r',ls=':',label = 'Target')
#ax[0].plot(output_train,c='k',label = 'Model')
#ax[0].set_title('MSE Train = '+str(format(MSE_train,".3E")))
#ax[0].legend()
#
### Plot the test data
#ax[1].plot(np.linspace(0,len(output_test),len(output_test)),df_data2['Target'],c='r',ls=':',label = 'Target')
#ax[1].plot(output_test,c='k',label = 'Model')
#ax[1].set_title('MSE Test = '+str(format(MSE_test,".3E")))
#ax[1].legend()

## Plot previous step
#ax[2].plot(y_test,c='r',ls=':')
#ax[2].plot(x_test['Output0'],c='k')
#ax[2].set_title('MSE Previous Step = '+str(format(MSE_prev_step,".3E")))
#ax[2].legend()
plt.legend()
plt.show()