from scipy.stats import pearsonr
import torch
import numpy as np
import torch.utils.data as Data

import mne

noise_type = 'EMG'
#result_location = '/home/peng/Denoise/attention/result/EMG_MMNN_4(1)_2e-4_128' #Informer5_5 , MMNN_4 , simple_CNN , LSTM , fcNN , Novel_CNN , Complex_CNN
#model = torch.load(result_location + '/best_model.pth')
#model.cpu()
file_location = '/home/peng/Denoise/attention/data/' 
noiseEEG_test = np.load( file_location + noise_type + '/noiseEEG_test.npy')  
EEG_test = np.load( file_location + noise_type + '/EEG_test.npy')  

num_test = noiseEEG_test.shape[0]

info = mne.create_info(ch_names=['AF7'],
                       ch_types=['eeg'] ,
                       sfreq=256)

#noiseEEG_test = torch.tensor(noiseEEG_test)
#EEG_test = torch.tensor(EEG_test)
def RMS(x):
    return np.sqrt((x ** 2).sum() / len(x))
def RRMSE(out,y):
    return (RMS(out - y)) / RMS(y)

total_rrmse = 0
total_cc = 0

for i in range(num_test):

    raw = mne.io.RawArray(noiseEEG_test[i:i+1,:],info)
    y = EEG_test[i:i+1,:]
    if noise_type=='EOG':
        raw.filter(l_freq=12,h_freq=None,method='fir')   #高通滤波
    if noise_type=='EMG':
        raw.filter(l_freq=1,h_freq=40,method='fir')
    x, times = raw[:]
    x = x[0,:]
    y = y[0,:]
    rrmse = RRMSE(x,y)
    cc, p_value = pearsonr(x, y)
    total_rrmse = total_rrmse + rrmse
    total_cc = total_cc + cc
    print("第{}轮:".format(i+1))
average_rrmse = total_rrmse / num_test
average_cc = total_cc / num_test

print("测试集平均rrmse: {}".format(average_rrmse))
print("测试集平均cc: {}".format(average_cc))
