# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:57:46 2021

@author: Tong
"""

import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    #print(idx)
    return idx

def vortex_train_fit(x,a,b,c,d):
    return a + b*np.exp(-1*c*(x**d))

def ms_train_fit(x,b,c,d):
    return  c + (1-b*np.exp(d*x))

def ms_train_fit_popt(x,a,b,c):
    return  a+ b*x + c*x**2

def vortex_train_fit_popt(x,a,b,c):
    return a+ b*x + c*x**2

def remag_train_fit_popt(x,a,b,c):
    return a+c*x**2+b*x 

def sigmoid_fit(x,a,b,c,d):
    return a + (1/(1+np.exp(-x*c +d)))*b

def fit_training_data(foldername):
    file = foldername
    training_data = pd.read_excel(file,"train_comparison") 

    ## Extract relevant columns 
    df_18 = training_data.loc[:,['Loop_18','Ms_18']] 
    df_18p7 = training_data.loc[:,['Loop_18p7','Ms_18p7']]
    df_19p5 = training_data.loc[:,['Loop_19p5','Ms_19p5']]

    ## Crop the data
    df_18 = df_18.loc[(df_18.Loop_18 <= 30)]
    df_18p7 = df_18p7.loc[(df_18p7.Loop_18p7 <= 30)]
    df_19p5 = df_19p5.loc[(df_19p5.Loop_19p5 <= 30)]


    #Plot the curves
    plt.title("Training data")
    plt.plot(df_18['Loop_18'].values,df_18['Ms_18'].values)
    plt.plot(df_18p7['Loop_18p7'].values,df_18p7['Ms_18p7'].values)
    plt.plot(df_19p5['Loop_19p5'].values,df_19p5['Ms_19p5'].values)
    #plt.show()

    ## Fit the exponentials using vortex_train_fit - a+ b*exp(-1*c*(x**d)) - returns a,b,c,d
    popt_18, pcov_18 = curve_fit(vortex_train_fit,df_18['Loop_18'].values,df_18['Ms_18'].values,maxfev = 100000)
    popt_18p7, pcov_18p7 = curve_fit(vortex_train_fit,df_18p7['Loop_18p7'].values,df_18p7['Ms_18p7'].values,maxfev = 10000)
    popt_19p5, pcov_19p5 = curve_fit(vortex_train_fit,df_19p5['Loop_19p5'].values,df_19p5['Ms_19p5'].values,maxfev = 10000)
    ms_train_new_x = np.linspace(0,30,301)

    # Plot the fits
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_18))
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_18p7))
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_19p5))

    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()
    

    ## Make a list of the fitted values a,b,c,d
    fits = []    
    for i in range(len(popt_18)):
        fits.append([popt_18[i],popt_18p7[i],popt_19p5[i]])
    x = [18,18.7,19.5]

    
    ## Now we model the various fit values a,b,c,d so that we can interpolate to any field value
    popt_a_training, pcov_a = curve_fit(ms_train_fit_popt,x,fits[0],maxfev = 100000)
    popt_b_training, pcov_b = curve_fit(vortex_train_fit_popt,x,fits[1],maxfev = 100000)
    popt_c_training, pcov_c = curve_fit(vortex_train_fit_popt,x,fits[2],maxfev = 100000)
    popt_d_training, pcov_d = curve_fit(ms_train_fit_popt,x,fits[3],maxfev = 100000)

    # Plot the fits from above
    fits_x = np.linspace(18,19.5,100)
    #plt.figure()
    plt.title("Training data Coeff Fits")
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_a_training))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_b_training))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_c_training))
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_d_training))
    
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()
    
    return(popt_a_training,popt_b_training,popt_c_training,popt_d_training)
    
def fit_training_data_vortex(foldername):
    file = foldername
    training_data = pd.read_excel(file,"train_comparison")
    #print(training_data.head(5))
    df_18 = training_data.loc[:,['Loop_18','V_18']]
    #print(df_18)
    df_18p7 = training_data.loc[:,['Loop_18p7','V_18p7']]
    df_19p5 = training_data.loc[:,['Loop_19p5','V_19p5']]

    df_18 = df_18.loc[(df_18.Loop_18 <= 30)]
    df_18p7 = df_18p7.loc[(df_18p7.Loop_18p7 <= 30)]
    df_19p5 = df_19p5.loc[(df_19p5.Loop_19p5 <= 30)]
    #print(df_18p7)
    #print(df_19p5)
    plt.plot(df_18['Loop_18'].values,df_18['V_18'].values)
    plt.plot(df_18p7['Loop_18p7'].values,df_18p7['V_18p7'].values)
    plt.plot(df_19p5['Loop_19p5'].values,df_19p5['V_19p5'].values)
    #plt.show()
    param_bounds=([-np.inf,-np.inf,-np.inf,0.9],[np.inf,np.inf,np.inf,1.1])
    popt_18, pcov_18 = curve_fit(vortex_train_fit,df_18['Loop_18'].values,df_18['V_18'].values,maxfev = 100000,bounds=param_bounds)
    popt_18p7, pcov_18p7 = curve_fit(vortex_train_fit,df_18p7['Loop_18p7'].values,df_18p7['V_18p7'].values,maxfev = 100000,bounds=param_bounds)
    popt_19p5, pcov_19p5 = curve_fit(vortex_train_fit,df_19p5['Loop_19p5'].values,df_19p5['V_19p5'].values,maxfev = 100000,bounds=param_bounds)
    ms_train_new_x = np.linspace(0,30,301)
    ms_train_new_x = np.linspace(0,30,301)
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_18))
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_18p7))
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_19p5))
    #plt.legend()
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()        
    fits = []
    for i in range(len(popt_18)):
        fits.append([popt_18[i],popt_18p7[i],popt_19p5[i]])
    x = [18,18.7,19.5]
    for i in range(len(fits)):
        plt.plot(x,fits[i],label = str(i))
    plt.legend()
    

    popt_a_training, pcov_a = curve_fit(ms_train_fit_popt,x,fits[0],maxfev = 100000)
    popt_b_training, pcov_b = curve_fit(vortex_train_fit_popt,x,fits[1],maxfev = 100000)
    popt_c_training, pcov_c = curve_fit(vortex_train_fit_popt,x,fits[2],maxfev = 100000)
    popt_d_training, pcov_d = curve_fit(ms_train_fit_popt,x,fits[3],maxfev = 100000)
    fits_x = np.linspace(18,19.5,100)
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_a_training))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_b_training))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_c_training))
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_d_training))
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()
    
    return(popt_a_training,popt_b_training,popt_c_training,popt_d_training)
    
def fit_training_frequency_data(foldername):
    file = foldername
    training_data = pd.read_excel(file,"train_comparison")

    #Extract relevent columns
    df_18 = training_data.loc[:,['Loop_18','Ms_f_18']]
    df_18p7 = training_data.loc[:,['Loop_18p7','Ms_f_18p7']]
    df_19p5 = training_data.loc[:,['Loop_19p5','Ms_f_19p5']]

    #crop data
    df_18 = df_18.loc[(df_18.Loop_18 <= 30)]
    df_18p7 = df_18p7.loc[(df_18p7.Loop_18p7 <= 30)]
    df_19p5 = df_19p5.loc[(df_19p5.Loop_19p5 <= 30)]

    #plot values of data
    plt.plot(df_18['Loop_18'].values,df_18['Ms_f_18'].values)
    plt.plot(df_18p7['Loop_18p7'].values,df_18p7['Ms_f_18p7'].values)
    plt.plot(df_19p5['Loop_19p5'].values,df_19p5['Ms_f_19p5'].values)
    #plt.show()
    
    #fit data with vortex train fit equation
    param_bounds=([5,0,0,0],[8,1,1,3])
    popt_18, pcov_18 = curve_fit(vortex_train_fit,df_18['Loop_18'].values,df_18['Ms_f_18'].values,maxfev = 100000,p0=[6,1,1,0.2],bounds=param_bounds)
    popt_18p7, pcov_18p7 = curve_fit(vortex_train_fit,df_18p7['Loop_18p7'].values,df_18p7['Ms_f_18p7'].values,maxfev = 100000,p0=[6,1,1,0.2],bounds=param_bounds)
    popt_19p5, pcov_19p5 = curve_fit(vortex_train_fit,df_19p5['Loop_19p5'].values,df_19p5['Ms_f_19p5'].values,maxfev = 100000,p0=[5,1,1,0.2],bounds=param_bounds)
    
    #plot fits
    ms_train_new_x = np.linspace(0,30,301)
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_18))
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_18p7))
    plt.plot(ms_train_new_x,vortex_train_fit(ms_train_new_x,*popt_19p5))
    #plt.legend()
    
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()    
        
    fits = []
    for i in range(len(popt_18)):
        fits.append([popt_18[i],popt_18p7[i],popt_19p5[i]])
    x = [18,18.7,19.5]
    
    
    for i in range(len(fits)):
        plt.plot(x,fits[i],label = str(i))
    plt.legend()
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()

    popt_a_training_frequency, pcov_a = curve_fit(ms_train_fit_popt,x,fits[0],maxfev = 100000)
    popt_b_training_frequency, pcov_b = curve_fit(vortex_train_fit_popt,x,fits[1],maxfev = 100000)
    popt_c_training_frequency, pcov_c = curve_fit(vortex_train_fit_popt,x,fits[2],maxfev = 100000)
    popt_d_training_frequency, pcov_d = curve_fit(ms_train_fit_popt,x,fits[3],maxfev = 100000)
    
    fits_x = np.linspace(18,19.5,100)
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_a_training_frequency))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_b_training_frequency))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_c_training_frequency))
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_d_training_frequency))
    plt.show()
    
    field_value = 18
    popt_f_a = ms_train_fit_popt(field_value,*popt_a_training_frequency)
    popt_f_b = ms_train_fit_popt(field_value,*popt_b_training_frequency)
    popt_f_c = ms_train_fit_popt(field_value,*popt_c_training_frequency)
    popt_f_d = ms_train_fit_popt(field_value,*popt_d_training_frequency)
    
    x = np.linspace(0,30,61)
    fields = vortex_train_fit(x,popt_f_a,popt_f_b,popt_f_c,popt_f_d)
    plt.plot(x,fields)
    
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()
        
    return(popt_a_training_frequency,popt_b_training_frequency,popt_c_training_frequency,popt_d_training_frequency)
    
def fit_remag_data(foldername):
    file = foldername
    training_data = pd.read_excel(file,"inch_up")

    #extract data
    df_5 = training_data.loc[:,['field_5','ms_5']]
    df_10 = training_data.loc[:,['field_10','ms_10']]
    df_30 = training_data.loc[:,['field_30','ms_30']]

    #crop extracted data
    df_5 = df_5.loc[(df_5.field_5 >= 19)]
    df_10 = df_10.loc[(df_10.field_10 >= 19)]
    df_30 = df_30.loc[(df_30.field_30 >= 19)]

    #plot data
    #plt.figure()
    plt.title("Remag data Fits")
    plt.plot(df_5['field_5'].values,df_5['ms_5'].values,"x")
    plt.plot(df_10['field_10'].values,df_10['ms_10'].values,"x")
    plt.plot(df_30['field_30'].values,df_30['ms_30'].values,"x")
    #plt.show()
    
    #create fit for data with sigmoid fit
    p0 = [0.5,0.5,1,21]
    popt_5, pcov_5 = curve_fit(sigmoid_fit,df_5['field_5'].values,df_5['ms_5'].values,p0=p0,maxfev = 100000)
    popt_10, pcov_10 = curve_fit(sigmoid_fit,df_10['field_10'].values,df_10['ms_10'].values,p0=p0,maxfev = 10000)
    popt_30, pcov_30 = curve_fit(sigmoid_fit,df_30['field_30'].values,df_30['ms_30'].values,p0=p0,maxfev = 10000)
    ms_train_new_x = np.linspace(19,25,301)

    #plot sigmoid fit
    plt.plot(ms_train_new_x,sigmoid_fit(ms_train_new_x,*popt_5),label = str(popt_5))
    plt.plot(ms_train_new_x,sigmoid_fit(ms_train_new_x,*popt_10),label = str(popt_10))
    plt.plot(ms_train_new_x,sigmoid_fit(ms_train_new_x,*popt_30),label = str(popt_30))
    #plt.legend()
    
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()        
        
    fits = []
    for i in range(len(popt_5)):
        fits.append([popt_5[i],popt_10[i],popt_30[i]])
    x = [5,10,30]

    #fit sigmoid coefficients for different applied fields
    popt_a_remag, pcov_a = curve_fit(remag_train_fit_popt,x,fits[0],maxfev = 100000)
    popt_b_remag, pcov_b = curve_fit(remag_train_fit_popt,x,fits[1],maxfev = 100000)
    popt_c_remag, pcov_c = curve_fit(remag_train_fit_popt,x,fits[2],maxfev = 100000)
    popt_d_remag, pcov_d = curve_fit(remag_train_fit_popt,x,fits[3],maxfev = 100000)
    
    fits_x = np.linspace(0,30,100)
    #fits_x = np.linspace(18,19.5,100)
    #plt.figure()
    
    plt.title("Remag data Coeff Fits")
    for i in range(0, len(fits)):
        plt.plot(x, fits[i],'x',label = f'{i}')
        
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_a_remag), label = "a")
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_b_remag), label = "b")
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_c_remag), label = "c")
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_d_remag), label = "d")
    plt.legend()
    
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()

    #create sigmoid fit for all different applied fields
    df_remag_look_up = pd.DataFrame()
    field_value = np.linspace(0,30,61)
    
    for i in range(len(field_value)):
        popt_a = ms_train_fit_popt(field_value[i],*popt_a_remag)
        popt_b = ms_train_fit_popt(field_value[i],*popt_b_remag)
        popt_c = ms_train_fit_popt(field_value[i],*popt_c_remag)
        popt_d = ms_train_fit_popt(field_value[i],*popt_d_remag)

        #plt.plot(ms_train_new_x,sigmoid_fit(ms_train_new_x,popt_a,popt_b,popt_c,popt_d),label = str(field_value[i]))
        #print(popt_a,popt_b,popt_c,popt_d)
        df_remag_look_up[field_value[i]] = sigmoid_fit(ms_train_new_x,popt_a,popt_b,popt_c,popt_d)
        
    df_remag_look_up = df_remag_look_up.T
    #print(df_remag_look_up)
    #plt.legend()
    #plt.show()
    
    return(popt_a_remag,popt_b_remag,popt_c_remag,popt_d_remag,df_remag_look_up)

def fit_remag_data_vortex(foldername):
    file = foldername
    training_data = pd.read_excel(file,"inch_up")
    #print(training_data.head(5))
    df_5 = training_data.loc[:,['field_5','v_5']]
    #print(df_18)
    df_10 = training_data.loc[:,['field_10','v_10']]
    df_30 = training_data.loc[:,['field_30','v_30']]

    df_5 = df_5.loc[(df_5.field_5 >= 19)]
    df_10 = df_10.loc[(df_10.field_10 >= 19)]
    df_30 = df_30.loc[(df_30.field_30 >= 19)]

    plt.plot(df_5['field_5'].values,df_5['v_5'].values)
    plt.plot(df_10['field_10'].values,df_10['v_10'].values)
    plt.plot(df_30['field_30'].values,df_30['v_30'].values)
    #plt.show()
    p0 = [0.5,0.5,1,21]
    popt_5, pcov_5 = curve_fit(sigmoid_fit,df_5['field_5'].values,df_5['v_5'].values,p0=p0,maxfev = 100000)
    popt_10, pcov_10 = curve_fit(sigmoid_fit,df_10['field_10'].values,df_10['v_10'].values,p0=p0,maxfev = 10000)
    popt_30, pcov_30 = curve_fit(sigmoid_fit,df_30['field_30'].values,df_30['v_30'].values,p0=p0,maxfev = 10000)
    ms_train_new_x = np.linspace(19,25,301)

    plt.plot(ms_train_new_x,sigmoid_fit(ms_train_new_x,*popt_5),label = str(popt_5))
    plt.plot(ms_train_new_x,sigmoid_fit(ms_train_new_x,*popt_10),label = str(popt_10))
    plt.plot(ms_train_new_x,sigmoid_fit(ms_train_new_x,*popt_30),label = str(popt_30))
    #plt.legend()
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()        
    fits = []
    for i in range(len(popt_5)):
        fits.append([popt_5[i],popt_10[i],popt_30[i]])
    x = [5,10,30]

    popt_a_remag_vortex, pcov_a = curve_fit(remag_train_fit_popt,x,fits[0],maxfev = 100000)
    popt_b_remag_vortex, pcov_b = curve_fit(remag_train_fit_popt,x,fits[1],maxfev = 100000)
    popt_c_remag_vortex, pcov_c = curve_fit(remag_train_fit_popt,x,fits[2],maxfev = 100000)
    popt_d_remag_vortex, pcov_d = curve_fit(remag_train_fit_popt,x,fits[3],maxfev = 100000)
    fits_x = np.linspace(0,30,100)
    
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_a_remag_vortex))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_b_remag_vortex))
    plt.plot(fits_x,vortex_train_fit_popt(fits_x,*popt_c_remag_vortex))
    plt.plot(fits_x,ms_train_fit_popt(fits_x,*popt_d_remag_vortex))
    if show_plots==True:
        plt.show()
    if show_plots==False:
        plt.close()

    df_remag_look_up_vortex = pd.DataFrame()
    field_value = np.linspace(0,30,61)
    for i in range(len(field_value)):
        popt_a = ms_train_fit_popt(field_value[i],*popt_a_remag_vortex)
        popt_b = ms_train_fit_popt(field_value[i],*popt_b_remag_vortex)
        popt_c = ms_train_fit_popt(field_value[i],*popt_c_remag_vortex)
        popt_d = ms_train_fit_popt(field_value[i],*popt_d_remag_vortex)

        df_remag_look_up_vortex[field_value[i]] = sigmoid_fit(ms_train_new_x,popt_a,popt_b,popt_c,popt_d)
    df_remag_look_up_vortex = df_remag_look_up_vortex.T

    return(popt_a_remag_vortex,popt_b_remag_vortex,popt_c_remag_vortex,popt_d_remag_vortex,df_remag_look_up_vortex)
    
def Vortex_field_new(ms_amp,ms_freq,v_amp,field_value, fit_ms_amp, fit_ms_freq, fit_v_amp):
    
    popt_a_training,popt_b_training,popt_c_training,popt_d_training = fit_ms_amp
    popt_a_training_frequency,popt_b_training_frequency,popt_c_training_frequency,popt_d_training_frequency = fit_ms_freq
    popt_a_training_vortex,popt_b_training_vortex,popt_c_training_vortex,popt_d_training_vortex = fit_v_amp
    
    ## Get the appropriate parameters for the fits
    popt_a = ms_train_fit_popt(field_value,*popt_a_training)
    popt_b = ms_train_fit_popt(field_value,*popt_b_training)
    popt_c = ms_train_fit_popt(field_value,*popt_c_training)
    popt_d = ms_train_fit_popt(field_value,*popt_d_training)
    x = np.linspace(0,30,61)


    fields = vortex_train_fit(x,popt_a,popt_b,popt_c,popt_d)
    index = find_nearest(fields,ms_amp)

    effective_loop = x[index]
    ms_amp_new = vortex_train_fit(effective_loop+1,popt_a,popt_b,popt_c,popt_d)
    
    ms_diff = ms_amp_new - ms_amp
    
    #print(ms_amp_new)
    popt_f_a = ms_train_fit_popt(field_value,*popt_a_training_frequency)
    popt_f_b = ms_train_fit_popt(field_value,*popt_b_training_frequency)
    popt_f_c = ms_train_fit_popt(field_value,*popt_c_training_frequency)
    popt_f_d = ms_train_fit_popt(field_value,*popt_d_training_frequency)
    #print(popt_f_a,popt_f_b,popt_f_c,popt_f_d)
    x = np.linspace(0,30,61)
    fields = vortex_train_fit(x,popt_f_a,popt_f_b,popt_f_c,popt_f_d)
    #plt.plot(fields)
    #plt.show()
    index = find_nearest(fields,ms_freq)

    effective_loop = x[index]
    ms_freq_new = vortex_train_fit(effective_loop+1,popt_f_a,popt_f_b,popt_f_c,popt_f_d)

    popt_vf_a = ms_train_fit_popt(field_value,*popt_a_training_vortex)
    popt_vf_b = ms_train_fit_popt(field_value,*popt_b_training_vortex)
    popt_vf_c = ms_train_fit_popt(field_value,*popt_c_training_vortex)
    popt_vf_d = ms_train_fit_popt(field_value,*popt_d_training_vortex)
    #print(popt_f_a,popt_f_b,popt_f_c,popt_f_d)
    x = np.linspace(0,30,61)
    fields = vortex_train_fit(x,popt_vf_a,popt_vf_b,popt_vf_c,popt_vf_d)

    index = find_nearest(fields,v_amp)

    effective_loop = x[index]

    v_amp_new = vortex_train_fit(effective_loop+1,popt_vf_a,popt_vf_b,popt_vf_c,popt_vf_d)

    return(ms_amp_new,ms_freq_new,v_amp_new,ms_diff)

def Remag_field_new(ms_amp,v_amp,field_value, prev_field, fit_ms_amp, fit_v_amp):
    
    popt_a_remag,popt_b_remag,popt_c_remag,popt_d_remag,df_remag_look_up = fit_ms_amp
    popt_a_remag_vortex,popt_b_remag_vortex,popt_c_remag_vortex,popt_d_remag_vortex, df_remag_look_up_vortex = fit_v_amp
    
    next_value = field_value + (field_value - prev_field)
    prev_field = field_value
    
    if prev_field < 19:
        prev_field = 19
        
    field = (prev_field- 19)*300/6
    field = round(field)
    index = find_nearest(df_remag_look_up[field].tolist(),ms_amp)

    index = index/2
    #print(index)
    #plt.scatter(prev_field,ms_amp)
    #print(prev_value)

    popt_a = remag_train_fit_popt(index,*popt_a_remag)
    popt_b = remag_train_fit_popt(index,*popt_b_remag)
    popt_c = remag_train_fit_popt(index,*popt_c_remag)
    popt_d = remag_train_fit_popt(index,*popt_d_remag)

    #x = np.linspace(19,25,61)
    #fields = sigmoid_fit(x,popt_a,popt_b,popt_c,popt_d)
    
    
    new_ms = sigmoid_fit(field_value,popt_a,popt_b,popt_c,popt_d)

    if new_ms < ms_amp:
        new_ms = ms_amp
        
    ms_diff = new_ms - ms_amp
    
    if ms_diff == 0:
        print(ms_amp, field_value, prev_field)

    if prev_field<19:
        prev_field = 19
    field = (prev_field - 19)*300/6
    field = round(field)

    index = find_nearest(df_remag_look_up_vortex[field].tolist(),v_amp)
    #index = df_remag_look_up.iloc[(df_remag_look_up[field]-ms_amp).abs().argsort()[:2]]
    
    index = index/2
    #print(index)
    #plt.scatter(prev_field,ms_amp)

    popt_a = remag_train_fit_popt(index,*popt_a_remag_vortex)
    popt_b = remag_train_fit_popt(index,*popt_b_remag_vortex)
    popt_c = remag_train_fit_popt(index,*popt_c_remag_vortex)
    popt_d = remag_train_fit_popt(index,*popt_d_remag_vortex)

    #x = np.linspace(19,25,61)
    #fields = sigmoid_fit(x,popt_a,popt_b,popt_c,popt_d)
    new_v = sigmoid_fit(field_value,popt_a,popt_b,popt_c,popt_d)

#    if new_v > v_amp:
#        new_v = v_amp

    return(new_ms,new_v, ms_diff)
    
def extract_data(filepath,column_name, sheet = 'LP_Scale0'):
    path = os.path.join(filepath,'all_data.xlsx') ## Path to the datafile
    data = pd.read_excel(path,sheet_name =sheet) ## Load data into a pandas dataframe
    ms_peak_freq = data[column_name].tolist() #209
    v_peak_freq = data['IQ105'].tolist()
    H_app = data['H_app2'].tolist()
    return H_app, ms_peak_freq, v_peak_freq

def extract_peak(filepath,sheet = 'LP_Scale0'):
    path = os.path.join(filepath,'all_data.xlsx') ## Path to the datafile
    data = pd.read_excel(path,sheet_name =sheet)
    ms_peak = data.iloc[:,180:230].max(axis=1)
    print(ms_peak)
    ms_peak = ms_peak.tolist()
    return ms_peak

def normalise_data(data):
    max_index = np.argmax([abs(x) for x in data])
    max_value = data[max_index]
    data = [x/max_value for x in data]
    return data
    
def vortex_remag_weight(field, cutoff, min_field, max_field, a):
    if field < cutoff:
        vortex_weight = 1- (0.5*((field - min_field)/(cutoff - min_field))**a)
        remag_weight = 1 - vortex_weight
    elif field >= cutoff:
        remag_weight  = 1 - (0.5*((max_field - field)/(max_field - cutoff))**a)
        vortex_weight = 1-remag_weight
    return vortex_weight, remag_weight
    
def vortex_remag_weight2(field, cutoff, min_field, max_field):
    vortex_weight = np.exp(-5*(field - min_field)/(cutoff - min_field))
    remag_weight = np.exp(-2*(max_field - field)/(max_field - cutoff))
    return vortex_weight, remag_weight

def model_data():
    file = r"C:\Users\Tong\Desktop\Msci\Files_for_MSci_students\Samples\HDS_GS\Long_sweeps\Sin\all_data.xlsx"
    training_data = pd.read_excel(file,"LP_Scale0") 
    ms_data  = training_data.loc[:,['H_app2','IQ209','IQ208','IQ207','IQ210','IQ211','IQ212']] #IQ216 for random
    #print(ms_data)
    return ms_data
    
def model_step(ms_data, ms_amp, field_value):
    app_field = ms_data['H_app2'].values
    app_field = [x/10 for x in app_field]
    max_app_field = max(app_field)
    #app_field = [x/max_app_field for x in app_field]
    obs_peak = ms_data['IQ209'].values
    obs_peak = []
    for i in ['IQ209','IQ208','IQ207','IQ210','IQ211','IQ212']:
        for j in range(0, len(ms_data[f"{i}"].values)):
            obs_peak.append(ms_data[f"{i}"][j])
    #print(obs_peak)
    max_obs_peak = max(obs_peak)
    #scaled_obs_peak = [x/max_obs_peak for x in obs_peak]
    diff_array = []
    for i in range(0,len(app_field)):
        field_diff = app_field[i]- field_value
        ms_diff = obs_peak[i] - ms_amp
        diff_sqr = (field_diff/max_app_field)**2 + (ms_diff/max_obs_peak)**2
        diff_array.append(diff_sqr)
        #print(field_diff,ms_diff,diff_sqr)
    #print(diff_array)
    index = np.argmin(diff_array)
    if index < len(app_field)-1:
        new_ms = obs_peak[index+1] - obs_peak[index]
    else:
        new_ms = 0
    return new_ms
 
def run_model(input_field,max_field,min_field,vortex_remag_cutoff,fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp,fit_remag_ms,fit_remag_v,ms_data_output):
    
    a_optimum = 2
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
    for i in range(0, len(input_field)):
        input_field[i]
        if input_field[i] > max_field:
            input_field[i] = max_field
        if input_field[i] < min_field:
            input_field[i] = min_field
        if input_field[i] < vortex_remag_cutoff:
            new_peak = Vortex_field_new(ms, f, v, input_field[i], fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp)
            ms, f, v = new_peak[0:3]
            output_ms_amp.append(new_peak[0])
            output_ms_freq.append(new_peak[1])
            output_v_amp.append(new_peak[2])
        elif input_field[i] >= vortex_remag_cutoff:
            new_peak = Remag_field_new(ms,v,input_field[i],input_field[i-1], fit_remag_ms, fit_remag_v)
            ms, v = new_peak[0:2]
            output_ms_amp.append(new_peak[0])
            output_v_amp.append(new_peak[1])
        vortex_diff = Vortex_field_new(ms_test, f, v, input_field[i], fit_training_ms_amp, fit_training_ms_freq, fit_training_v_amp)[-1]
        remag_diff = Remag_field_new(ms_test,v,input_field[i],input_field[i-1], fit_remag_ms, fit_remag_v)[-1]
        vortex_weight, remag_weight = vortex_remag_weight(input_field[i],vortex_remag_cutoff,min_field, max_field,a_optimum)
        #print(vortex_weight, remag_weight)
        ms_test += vortex_weight*vortex_diff + remag_weight*remag_diff
        output_ms_weighted.append(ms_test)
        
        
    
    output_ms_amp = normalise_data(output_ms_amp)
    
#    mse_sum = 0
#    for i in range(0, len(output_ms_weighted)):
#        mse_sum += (output_ms_weighted[i] - ms_data_output[i])**2
        
#    print(mse_sum)
        
#    output_ms_weighted = normalise_data(output_ms_weighted)
#    
#    mse_data = []
#    for i in range(0, len(output_ms_amp)):
#        mse_data.append((output_ms_amp[i] - ms_data_output[i])**2)
#    
#    mse_weighted_data = []
#    for i in range(0, len(output_ms_weighted)):
#        mse_weighted_data.append((output_ms_weighted[i] - ms_data_output[i])**2)
        
    
    
    fig,ax = plt.subplots(5,1)
    ax[0].plot(input_field)
    ax[0].set_title("Input field")
    
    #ax[1].plot(ms_data_output)
    ax[1].set_title("Experimental Output")
    
    #ax[2].plot(ms_data_output, label = "Experimental")
    ax[2].plot(output_ms_amp, label = "Computational")
    ax[2].set_title("Computational Output")
    ax[2].legend()
    
    #ax[3].plot(ms_data_output, label = "Experimental")
    ax[3].plot(output_ms_weighted, label = "Computational")
    ax[3].set_title("Weighted Computational Output")
    ax[3].legend()
    
#    ax[4].plot(mse_data, label = "Not weighted")
#    ax[4].plot(mse_weighted_data, label = "weighted")
#    ax[4].set_title("Error Squared")
#    ax[4].legend()
    
    plt.show()
    
    return input_field, output_ms_weighted
    
show_plots = False
