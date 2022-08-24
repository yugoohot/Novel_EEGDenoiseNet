# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter 
from model import *     
from torchsummary import summary
from network import *
import time
from errormodel import *


#-----------------------------设置参数------------------------------------
batch_size = 256
noise_type = 'EOG'
datanum = 512
result_location = '/home/peng/Denoise/attention/result/MMNN_4_4e-4_256' 
model = MMNN_4(T=datanum,c=32,k=25)  # simple_CNN , fcNN , MMNN_4(T=datanum,c=32,k=25) , Informer5_5 , Novel_CNN , Complex_CNN
epoch = 200
test = True
learning_rate = 4e-4

model = model.cuda(0)
summary(model,input_size=(1, datanum), batch_size=batch_size)
#-----------------------------读取数据-------------------------------------
file_location = '/home/peng/Denoise/attention/data/'                     

noiseEEG_train = np.load( file_location + noise_type + '/noiseEEG_train.npy')
EEG_train = np.load( file_location + noise_type + '/EEG_train.npy')  
noiseEEG_val = np.load( file_location + noise_type + '/noiseEEG_val.npy')  
EEG_val = np.load( file_location + noise_type + '/EEG_val.npy')  

num_train = noiseEEG_train.shape[0]
num_val = noiseEEG_val.shape[0]

print("训练集数据",noiseEEG_train.shape)
print("验证集数据",noiseEEG_val.shape)

noiseEEG_train = torch.tensor(noiseEEG_train)
EEG_train = torch.tensor(EEG_train)
noiseEEG_val = torch.tensor(noiseEEG_val)
EEG_val = torch.tensor(EEG_val)

# (batch_size,512) --> (batch_size,1,512)
noiseEEG_train = noiseEEG_train.unsqueeze(1)
EEG_train = EEG_train.unsqueeze(1)
EEG_val = EEG_val.unsqueeze(1)
noiseEEG_val = noiseEEG_val.unsqueeze(1)


train_dataset = Data.TensorDataset(noiseEEG_train, EEG_train)
train_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train_dataset,       # torch TensorDataset format
    batch_size=batch_size,       # mini batch size
    shuffle=True,                # 要不要打乱数据 (打乱比较好)
    num_workers=2,               # 多线程来读数据
)

val_dataset = Data.TensorDataset(noiseEEG_val, EEG_val)
val_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=val_dataset,         # torch TensorDataset format
    batch_size=batch_size,       # mini batch size
    shuffle=True,                # 要不要打乱数据 (打乱比较好)
    num_workers=2,               # 多线程来读数据
)
#batch_num = math.ceil(noiseEEG_train.shape[0]/batch_size)


#-----------------------------准备训练-------------------------------------
# 创建损失函数
loss_fn = nn.MSELoss(reduction='mean')
loss_fn = loss_fn.cuda(0)
loss_fn.requires_grad_(True)
# 优化器      
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate,alpha=0.9)    
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9), eps=5e-5)          
#TensorBoard可视化
writer = SummaryWriter(result_location)

val_loss_list = [1]
train_loss_list = [1] #结束后输出txt
#初始化
'''for m in model.modules():
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight,mode='fan_in', nonlinearity='relu')'''
from tqdm import tqdm
#model.train()
log = ['average_train_loss','average_val_loss','timeconsume']
for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))
    starttime = time.time()#记录开始时间
    # 训练
    #model.train()
    total_train_loss = 0
    for i, (x, y) in enumerate(tqdm(train_loader)):
    #for data in train_loader:
        #x, y = data
        x = x.to(torch.float32).cuda(0)
        y = y.to(torch.float32).cuda(0)

        outputs = model(x)
        
        train_loss = loss_fn(outputs, y)   

        optimizer.zero_grad()               # 梯度清零
        train_loss.backward()                     # 反向传播
        optimizer.step()                    # 对参数进行优化
        #画loss
        total_train_loss = total_train_loss + train_loss
    average_train_loss = total_train_loss / num_train
    print("训练集平均loss: {}".format(average_train_loss))
    writer.add_scalar("average_train_loss", average_train_loss, i) #TnesorBoard参数显示


    # 验证
    #model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            val_x, val_y = data
            val_x = val_x.to(torch.float32).cuda(0)
            val_y = val_y.to(torch.float32).cuda(0)

            outputs = model(val_x)
            val_loss = loss_fn(outputs, val_y)            # 这里的 loss 只是一部分数据(data) 在网络模型上的损失
            total_val_loss = total_val_loss + val_loss    # 整个测试集的loss

        average_val_loss = total_val_loss / num_val

    print("验证集平均loss: {}".format(average_val_loss))
    writer.add_scalar("average_val_loss", average_val_loss, i) #TnesorBoard参数显示


    endtime = time.time()#记录结束时间
    consumetime = float(endtime - starttime)
    print('本轮耗时：',consumetime)
    #timelist.append(consumetime)

    #保存模型
    
    if average_val_loss < min(val_loss_list):
        saved_model = model
        torch.save(saved_model, result_location + "/best_model.pth")
        print("模型已保存")
    val_loss_list.append(average_val_loss)


    #保存log
    log.append(float(average_train_loss))
    log.append(float(average_val_loss))
    log.append(consumetime)
    with open(result_location + "/log.csv",'a',encoding='utf-8') as f:
        f.write('\n' + str(log))
    log = []

writer.close() #Tensorboard关闭
#np.savetxt(result_location + '//train_loss_list.txt',train_loss_list)
#np.savetxt(result_location + '//val_loss_list.txt',val_loss_list) #保存历史验证loss


#---------------------------开始测试集数据处理-------------------------------
'''if test == True:
    num_test = noiseEEG_test.shape[0]

    print("测试集数据",noiseEEG_test.shape)

    noiseEEG_test = torch.tensor(noiseEEG_test)
    EEG_test = torch.tensor(EEG_test)
    # (batch_size,512) --> (batch_size,1,512)
    EEG_test = EEG_test.unsqueeze(1)
    noiseEEG_test = noiseEEG_test.unsqueeze(1)

    test_dataset = Data.TensorDataset(noiseEEG_test, EEG_test)
    test_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=test_dataset,       # torch TensorDataset format
        batch_size=batch_size,       # mini batch size
        shuffle=True,                # 要不要打乱数据 (打乱比较好)
        num_workers=2,               # 多线程来读数据
    )

    # 开始测试
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            test_x, test_y = data
            test_x = test_x.to(torch.float32)
            test_y = test_y.to(torch.float32)
            outputs = model(test_x)
            loss = loss_fn(outputs, test_y)             # 这里的 loss 只是一部分数据(data) 在网络模型上的损失
            total_test_loss = total_test_loss + loss    # 整个测试集的loss

        average_test_loss = total_test_loss / num_test

    print("测试集平均loss: {}".format(average_test_loss))
    np.savetxt(result_location + '//test_loss.txt', average_test_loss) #保存测试loss'''
