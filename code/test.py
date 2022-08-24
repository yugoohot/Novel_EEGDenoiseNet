from scipy.stats import pearsonr
import torch
import numpy as np
import torch.utils.data as Data
import time
from tqdm import tqdm

batch_size = 100
noise_type = 'EOG'
result_location = '/home/peng/Denoise/attention/result/MMNN_4_4e-4_256' #Informer5_5 , MMNN_4 , simple_CNN , LSTM , fcNN , Novel_CNN , Complex_CNN
model = torch.load(result_location + '/best_model.pth')
model.cpu()
file_location = '/home/peng/Denoise/attention/data/' 
noiseEEG_test = np.load( file_location + noise_type + '/noiseEEG_test.npy')  
EEG_test = np.load( file_location + noise_type + '/EEG_test.npy')  

num_test = noiseEEG_test.shape[0]

noiseEEG_test = torch.tensor(noiseEEG_test)
EEG_test = torch.tensor(EEG_test)


# (batch_size,512) --> (batch_size,1,512)
noiseEEG_test = noiseEEG_test.unsqueeze(1)
EEG_test = EEG_test.unsqueeze(1)



test_dataset = Data.TensorDataset(noiseEEG_test, EEG_test)
test_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=test_dataset,       # torch TensorDataset format
    batch_size=batch_size,       # mini batch size
    shuffle=True,                # 要不要打乱数据 (打乱比较好)
    num_workers=2,               # 多线程来读数据
)

def RMS(x):
    return np.sqrt((x ** 2).sum() / len(x))
def RRMSE(out,y):
    return (RMS(out - y)) / RMS(y)

total_rrmse = 0
total_cc = 0
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
    
        test_x, test_y = data
        test_x = test_x.to(torch.float32).cpu()
        test_y = test_y.to(torch.float32).cpu()

        outputs = model(test_x)
        outputs = outputs.squeeze(1)
        
        test_y = test_y.squeeze(1)
        

        for i in range(outputs.shape[0]):
            x = outputs[i,:]
            y = test_y[i,:]
            rrmse = RRMSE(x,y)
            cc, p_value = pearsonr(x, y)
            total_rrmse = total_rrmse + rrmse
            total_cc = total_cc + cc    # 整个测试集的loss

    average_rrmse = total_rrmse / num_test
    average_cc = total_cc / num_test

print("测试集平均rrmse: {}".format(average_rrmse))
print("测试集平均cc: {}".format(average_cc))
