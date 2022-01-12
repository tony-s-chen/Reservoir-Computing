# -*- coding: utf-8 -*-
"""
@author: Tony
"""

import scipy.io
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
    '''
    extracts the data from an excel located at file_location
    sets the target to the target_bin column
    set number of previous input and output the data should extract
    
    returns: 
        df_data: dataframe for extracted data
        reservoir_output_names: list of column names of the df
    '''
    
    
    df=pd.read_excel(file_location,sheet_name)
    df_data = pd.DataFrame()
    
    # Assign the target point to be the frequency bin
    df_data['Target'] = df[target_bin]
    df_data['Current_input'] = df['H_app2']
    df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
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
                if i-j-1 >= 0:
                    df_data['Input'+str(j)].iloc[i] = df['H_app2'].iloc[i-j-1]
                else:
                    df_data['Input'+str(j)].iloc[i] = 0
            for j in range(prev_outputs):
                if i-j-1 >= 0:
                    df_data['Output'+str(j)].iloc[i] = df_data['Target'].iloc[i-j-1]
                else:
                    df_data['Output'+str(j)].iloc[i] = 0
    
    ## Remove nans - I have decided to keep the nans as they seem to improve accuracy of predictions
    #df_data = df_data.tail(len(df_data)-max([prev_outputs,prev_inputs]))

    #print(df_data)
    return df_data, reservoir_output_names


def data_extractor2(file_location, target_bin_num, prev_inputs, prev_outputs, sheet_name = 'LP_Scale0'):
    '''
    identicle to data_extractor() except this will extract the peak from frequencies around the target_bin_num
    instead of only looking at one frequency bin
    
    returns: 
        df_data: dataframe for extracted data
        reservoir_output_names: list of column names of the df
    '''
    
    df=pd.read_excel(file_location,sheet_name)
    df_data = pd.DataFrame()
    
    #extracts the peak from range of 25 around the target_bin
    ms_peak = df.iloc[:,(target_bin_num-25):(target_bin_num+25)].max(axis=1)
    print(ms_peak)

    # Assign the target point to be the frequency bin
    df_data['Target'] = ms_peak
    df_data['Current_input'] = df['H_app2']
    df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
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
                if i-j-1 >= 0:
                    df_data['Input'+str(j)].iloc[i] = df['H_app2'].iloc[i-j-1]
                else:
                    df_data['Input'+str(j)].iloc[i] = 0
            for j in range(prev_outputs):
                if i-j-1 >= 0:
                    df_data['Output'+str(j)].iloc[i] = df_data['Target'].iloc[i-j-1]
                else:
                    df_data['Output'+str(j)].iloc[i] = 0
    
    ## Remove nans
    #df_data = df_data.tail(len(df_data)-max([prev_outputs,prev_inputs]))

    #print(df_data)
    return df_data, reservoir_output_names

def data_extractor_MG(file_location, target_bin, prev_inputs, prev_outputs, sheet_name = 'LP_Scale0'):
    '''
    extracts the data from an excel located at file_location
    sets the target to the target_bin column
    set number of previous input and output the data should extract
    
    returns: 
        df_data: dataframe for extracted data
        reservoir_output_names: list of column names of the df
    '''
    
    
    df=pd.read_excel(file_location,sheet_name)
    print(df)
    df = df.drop(columns=['number', 'H_app'])
    df = df.rename(columns={'H_app2':'Current_input'})
    
    print(df)
    df2 = pd.read_excel(r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\\Mackey_Glass\all_data.xlsx",sheet_name)
    df_data = pd.DataFrame()
    
    # Assign the target point to be the frequency bin
    n = 5
    df_data["Target"] = df['Current_input'].shift(n)
    
    print(df_data)
    df_data = pd.concat([df_data, df],axis = 1, join = 'inner')
    #df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
    reservoir_output_names = []
    
    for col in df_data.columns:
        reservoir_output_names.append(col)
        
    print(reservoir_output_names)
    reservoir_output_names.pop(0)
    reservoir_output_names.pop(0)
    
    print(reservoir_output_names)
    ## Create empty df columns for input and outputs

    ## Include the current input as a parameter
    reservoir_output_names.append('Current_input')
    df_data = df_data.iloc[n:, :]
    print(df_data)

    #print(df_data)
    return df_data, reservoir_output_names

def empty_df(file_location, target_bin, prev_inputs, prev_outputs, sheet_name = 'LP_Scale0'):
    '''
    extracts the only the current_input and target data of the data
    no previous input/output columns are filled
    used to test on unknown input data with no given output
    
    returns: 
        df_data: dataframe for extracted data
        reservoir_output_names: list of column names of the df
    '''
    
    df=pd.read_excel(file_location,sheet_name)
    df_data = pd.DataFrame()

    
    # Assign the target point to be the frequency bin
    df_data['Current_input'] = df['H_app2']
    df_data['Target'] = df[target_bin]
    df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
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
    
    #attempt to give it the first 3 columns of previous data to provide an accurate starting point for the model
#    for i in range(3):
#            for j in range(prev_inputs):
#                if i-j-1 >= 0:
#                    df_data['Input'+str(j)].iloc[i] = df['H_app2'].iloc[i-j-1]
#                else:
#                    df_data['Input'+str(j)].iloc[i] = 0
#            for j in range(prev_outputs):
#                if i-j-1 >= 0:
#                    df_data['Output'+str(j)].iloc[i] = df_data['Target'].iloc[i-j-1]
#                else:
#                    df_data['Output'+str(j)].iloc[i] = 0

    return df_data, reservoir_output_names

def df_data(input_data, prev_inputs, prev_outputs, sim_output_data = 0, output_data = 0):
    
    df_data = pd.DataFrame()

    
    # Assign the target point to be the frequency bin
    df_data['Current_input'] = input_data
    df_data['Target'] = sim_output_data
    df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
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
    
    #attempt to give it the first 3 columns of previous data to provide an accurate starting point for the model
    for i in range(len(df_data)):
        for j in range(prev_inputs):
            if i-j-1 >= 0:
                df_data['Input'+str(j)].iloc[i] = df_data['Current_input'].iloc[i-j-1]
            else:
                df_data['Input'+str(j)].iloc[i] = 0
        for j in range(prev_outputs):
            if i-j-1 >= 0:
                df_data['Output'+str(j)].iloc[i] = df_data['Target'].iloc[i-j-1]
            else:
                df_data['Output'+str(j)].iloc[i] = 0
    
    df_data['Target'] = output_data
    df_data["Target"] = df_data["Target"].div(df_data["Target"].max())
    return df_data, reservoir_output_names
    
    
def mat_extract(file_location, prev_inputs, prev_outputs):
    mat = scipy.io.loadmat(file_location)
    df_data = pd.DataFrame()

    
    # Assign the target point to be the frequency bin
    df_data['Current_input'] = mat['val'][0]
    df_data['Target'] = 0
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
    
    ECG_range = max(df_data['Current_input']) - min(df_data['Current_input'])
    df_data['Current_input'] = df_data['Current_input']/ECG_range*50
    df_data['Current_input'] = df_data['Current_input'] - min(df_data['Current_input'])+180

    return df_data, reservoir_output_names

def regression(training_data, training_names, testing_data, testing_names, reg_type = "Ridge", show_plot = False):
    '''
    regresion model (based on code by Kilian)
    trains a regresion model on the provided training data and then tests on the testing data
    training_data: dataframe of the data to be used to train (output of data_extractor[0])
    training_names: list of column names in the dataframe (output of data_extractor[1])
    
    returns:
        reg: the trained regression model
        output_train: the predicted output from the model using the training data
        output_test: the predicted output from the model using the testing data
        MSE_train: mse of the training output from the experimental value
        MSE_test: mse of the testing output from the experimental value
    '''
    
    #decide which regression model, can add more to try
    b = 1e-4
    if reg_type == "Ridge":
        model=Ridge(alpha = b, fit_intercept = True,copy_X = True)
    elif reg_type == "Lasso":
        model=Lasso(alpha = b, fit_intercept = True,copy_X = True)
        
    reg = model.fit(training_data[training_names], training_data['Target'])
    
    ## Print the weights and intercept
#    print("ridge coeffs")
#    print(reg.coef_)
#    print(reg)
    
    ## Test on the test data

    output_train = reg.predict(training_data[training_names])
    output_test = reg.predict(testing_data[testing_names])
    
    MSE_train = mean_squared_error(training_data['Target'],output_train)
    MSE_test = mean_squared_error(testing_data['Target'],output_test)
    
    if show_plot:
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

def regression2(training_data, training_names, testing_data, testing_names,prev_inputs, prev_outputs, reg_type = "Ridge", show_plot = False):
    '''
    similar to regression() but this one will take in empty dataframes (dataframe which only have the current_input column) and iterate through to update and fill in the missing values of the dataframe for future steps
    however, the target column has been included in this to allow for MSE and experimental/prediction difference to be plotted
    
    returns:
        reg: the trained regression model
        output_train: the predicted output from the model using the training data
        output_test: the predicted output from the model using the testing data
        MSE_train: mse of the training output from the experimental value
        MSE_test: mse of the testing output from the experimental value
    '''
    
    #decide which regression model
    b = 1e-4
    if reg_type == "Ridge":
        model=Ridge(alpha = b, fit_intercept = True,copy_X = True)
    elif reg_type == "Lasso":
        model=Lasso(alpha = b, fit_intercept = True,copy_X = True)
        
    reg = model.fit(training_data[training_names], training_data['Target'])
    
    ## Print the weights and intercept
#    print("ridge coeffs")
#    print(reg.coef_)
#    print(reg)

    
    ## Test on the test data
    output_train = reg.predict(training_data[training_names])
    
    #iterates through the testing_data one row at a time and uses the output to update the future rows of data with necesary values
    for i in range(0,len(testing_data.index)):
        output_test = reg.predict(testing_data[testing_names].head(i+1))
        for j in range(prev_inputs):
            if i+j+1 < len(testing_data.index):
                testing_data['Input'+str(j)].iloc[i+j +1] = testing_data["Current_input"].head(i+1).iloc[-1]
        for j in range(prev_outputs):
            if i+j+1 < len(testing_data.index):
                testing_data['Output'+str(j)].iloc[i+j +1] = output_test[-1]
                
    
    MSE_train = mean_squared_error(training_data['Target'],output_train)
    MSE_test = mean_squared_error(testing_data['Target'],output_test)
    
    if show_plot:
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
