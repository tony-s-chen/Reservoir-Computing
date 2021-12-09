# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 01:03:07 2021

@author: Tong
"""

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd    
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import TimeSeriesSplit
import collections
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import math
from openpyxl import load_workbook
import sys
from sklearn.metrics import mean_squared_error

def data_extractor(file_location, target_bin, prev_inputs, prev_outputs, sheet_name = 'LP_Scale0'):
    
    df=pd.read_excel(file_location,sheet_name)
    df_data = pd.DataFrame()
    
    # Assign the target point to be the frequency bin
    df_data['Target'] = df[target_bin]
    df_data['Current_input'] = df['H_app2']
    reservoir_output_names = []
    
    ## Create empty df columns for input and outputs
    for i in range(prev_outputs):
    	df_data['Output'+str(i)] = 0
    	reservoir_output_names.append('Output'+str(i))
    
    for i in range(prev_inputs):
    	df_data['Input'+str(i)] = 0
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
    
    ## Remove nans
    df_data = df_data.tail(len(df_data)-max([prev_outputs,prev_inputs]))
    
    df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
    
    return df_data, reservoir_output_names

def regression(training_data, training_names, testing_data, testing_names, reg_type = "Ridge"):
    
    b = 1e-4
    if reg_type == "Ridge":
        model=Ridge(alpha = b, fit_intercept = True,copy_X = True)
    elif reg_type == "Lasso":
        model=Lasso(alpha = b, fit_intercept = True,copy_X = True)
        
    reg = model.fit(training_data[training_names], training_data['Target'])
    
    ## Print the weights and intercept
    print("ridge coeffs")
    print(reg.coef_)
    print(reg)
    
    ## Test on the test data

    output_train = reg.predict(training_data[training_names])
    output_test = reg.predict(testing_data[testing_names])
    
    MSE_train = mean_squared_error(training_data['Target'],output_train)
    MSE_test = mean_squared_error(testing_data['Target'],output_test)
    
    fig,ax = plt.subplots(2,1)
    
    ax[0].plot(np.linspace(0,len(output_train),len(output_train)), training_data['Target'],c='r',ls=':',label = 'Target')
    ax[0].plot(output_train,c='k',label = 'Model')
    ax[0].set_title('MSE Train = '+str(format(MSE_train,".3E")))
    ax[0].legend()
    
    ## Plot the test data
    ax[1].plot(np.linspace(0,len(output_test),len(output_test)),testing_data['Target'],c='r',ls=':',label = 'Target')
    ax[1].plot(output_test,c='k',label = 'Model')
    ax[1].set_title('MSE Test = '+str(format(MSE_test,".3E")))
    ax[1].legend()

    return reg, output_train, output_test, MSE_train, MSE_test
