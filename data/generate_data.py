# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
from torch import nn
import numpy as np

from data_prepare import *

#-----------------------------设置参数------------------------------------

noise_type = 'EOG'
combin_num = 20

#-----------------------------读取数据-------------------------------------
file_location = '/home/peng/Denoise/attention/data/'                     
if noise_type == 'EOG':
    EEG_all = np.load( file_location + 'EEG_all_epochs.npy')                              
    noise_all = np.load( file_location + 'EOG_all_epochs.npy') 
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(
    EEG_all = EEG_all, noise_all = noise_all, combin_num = combin_num, train_per = 0.8, noise_type = noise_type)

    np.save( file_location + "EOG1/noiseEEG_train.npy", noiseEEG_train)
    np.save( file_location + "EOG1/EEG_train.npy", EEG_train)
    np.save( file_location + "EOG1/noiseEEG_val.npy", noiseEEG_val)
    np.save( file_location + "EOG1/EEG_val.npy", EEG_val)
    np.save( file_location + "EOG1/noiseEEG_test.npy", noiseEEG_test)
    np.save( file_location + "EOG1/EEG_test.npy", EEG_test)

elif noise_type == 'EMG':
    EEG_all = np.load( file_location + 'EEG_all_epochs_512hz.npy')                              
    noise_all = np.load( file_location + 'EMG_all_epochs_512hz.npy') 

    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(
        EEG_all = EEG_all, noise_all = noise_all, combin_num = combin_num, train_per = 0.8, noise_type = noise_type)

    np.save( file_location + "EMG1/noiseEEG_train.npy", noiseEEG_train)
    np.save( file_location + "EMG1/EEG_train.npy", EEG_train)
    np.save( file_location + "EMG1/noiseEEG_val.npy", noiseEEG_val)
    np.save( file_location + "EMG1/EEG_val.npy", EEG_val)
    np.save( file_location + "EMG1/noiseEEG_test.npy", noiseEEG_test)
    np.save( file_location + "EMG1/EEG_test.npy", EEG_test)



