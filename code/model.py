import torch
from torch import nn
from wavelet_demo import *
from probattn import *
#from errormodel import *
from pytorch_wavelets import DWT1DForward, DWT1DInverse
'''input=torch.randn(50,49,512)
sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)
'''
# 搭建神经网络

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer1(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.tgt_mask = torch.triu(torch.full((datanum, datanum), float('-inf')), diagonal=0)
        self.C1 = nn.Conv1d(1, 512, 3, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(512)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.C2 = nn.Conv1d(512, 1, 3, stride=1, padding='same')

        self.PE = Pos_Encoding(d_model=512)
        self.T = nn.Transformer(d_model=512, nhead=4)
        self.z = torch.zeros((512,40,512))

    def forward(self, x, y):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        out = self.C1(x)
        out = self.B1(out)
        out = self.R1(out)
        out = self.drop1(out)
        # [batch_num, dim, len] ---> [len, batch_num, dim]
        out = out.transpose(2,0).transpose(2,1)
        y = y.transpose(2,0).transpose(2,1)
        # 位置编码
        out = self.PE(out)
        # 输入transformer
        y = y + self.z
        out = self.T(out, y, tgt_mask=self.tgt_mask)
        # [len, batch_num, dim] ---> [batch_num, dim, len]
        out = out.transpose(2,1).transpose(2,0)
        # [batch_num, dim, len] ---> [batch_num, 1, len]
        out = self.C2(out)

        return out

class Transformer1_1(nn.Module):#加了开始符号和结束符号
    def __init__(self,datanum) -> None:
        super().__init__()
        self.tgt_mask = torch.triu(torch.full((datanum+1, datanum+1), float('-inf')), diagonal=1)
        self.C1 = nn.Conv1d(1, 512, 3, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(512)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.C2 = nn.Conv1d(512, 1, 3, stride=1, padding='same')

        self.PE = Pos_Encoding(d_model=512)
        self.T = nn.Transformer(d_model=512, nhead=4)
        self.z = torch.zeros((512,40,512))
        self.start = torch.zeros((1,40,512))
        self.end = torch.ones((1,40,512))

    def forward(self, x, y):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        out = self.C1(x)
        out = self.B1(out)
        out = self.R1(out)
        out = self.drop1(out)
        # [batch_num, dim, len] ---> [len, batch_num, dim]
        out = out.transpose(2,0).transpose(2,1)
        y = y.transpose(2,0).transpose(2,1)
        # [len, batch_num, 1] ---> [len, batch_num, dim]
        y = y + self.z
        # 添加开始和结束标志：[len, batch_num, dim] ---> [len+1, batch_num, dim]
        y_input = torch.cat([self.start,y],0)
        out = torch.cat([out,self.end],0)
        # 位置编码
        y_input = self.PE(y_input)
        out = self.PE(out)
        # 输入transformer
        out = self.T(out, y_input, src_mask=self.tgt_mask, tgt_mask=self.tgt_mask)
        # [len+1, batch_num, dim] ---> [batch_num, dim, len+1]
        out = out.transpose(2,1).transpose(2,0)
        # [batch_num, dim, len+1] ---> [batch_num, 1, len+1]
        out = self.C2(out)

        return out

class Transformer2(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.C1 = nn.Conv1d(1, 4, 256, stride=1, padding='same')
        self.C11 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(16)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.C2 = nn.Conv1d(1, 4, 64, stride=1, padding='same')
        self.C21 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B2 = nn.BatchNorm1d(16)
        self.R2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.C3 = nn.Conv1d(1, 4, 16, stride=1, padding='same')
        self.C31 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B3 = nn.BatchNorm1d(16)
        self.R3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.C4 = nn.Conv1d(1, 4, 4, stride=1, padding='same')
        self.C41 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B4 = nn.BatchNorm1d(16)
        self.R4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)

        self.encodelayer = nn.TransformerEncoderLayer(d_model=64, nhead=8,batch_first=True,dropout=0.3)
        self.encoder = nn.TransformerEncoder(self.encodelayer, num_layers=3)

        self.F1 = nn.Linear(64,16)
        self.F2 = nn.Linear(16,4)
        self.F3 = nn.Linear(4,1)
       
    def forward(self, x):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        out1 = self.C1(x)
        out1 = self.C11(out1)
        out1 = self.B1(out1)
        out1 = self.R1(out1)
        out1 = self.drop1(out1)

        out2 = self.C2(x)
        out2 = self.C21(out2)
        out2 = self.B2(out2)
        out2 = self.R2(out2)
        out2 = self.drop2(out2)

        out3 = self.C3(x)
        out3 = self.C31(out3)
        out3 = self.B3(out3)
        out3 = self.R3(out3)
        out3 = self.drop3(out3)  

        out4 = self.C4(x)
        out4 = self.C41(out4)
        out4 = self.B4(out4)
        out4 = self.R4(out4)
        out4 = self.drop4(out4)  

        out = torch.cat([out1,out2,out3,out4],1)
        #[batch_num, dim, len] ---> [batch_num,len,dim]
        out = self.encoder(out.transpose(2,1))
        #[batch_num,len,dim] ---> [batch_num,1,len]
        out = (self.F3(self.F2(self.F1(out)))).transpose(1,2)
        return out

class Transformer3(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.C1 = nn.Conv1d(1, 4, 256, stride=1, padding='same')
        self.C11 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(16)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.C2 = nn.Conv1d(1, 4, 64, stride=1, padding='same')
        self.C21 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B2 = nn.BatchNorm1d(16)
        self.R2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.C3 = nn.Conv1d(1, 4, 16, stride=1, padding='same')
        self.C31 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B3 = nn.BatchNorm1d(16)
        self.R3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.C4 = nn.Conv1d(1, 4, 4, stride=1, padding='same')
        self.C41 = nn.Conv1d(4, 16, 5, stride=1, padding='same')
        self.B4 = nn.BatchNorm1d(16)
        self.R4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)

        self.encodelayer = nn.TransformerEncoderLayer(d_model=64, nhead=16,batch_first=True,dropout=0.3)
        self.encoder = nn.TransformerEncoder(self.encodelayer, num_layers=6)

        self.F1 = nn.Linear(64,16)
        self.F2 = nn.Linear(16,4)
        self.F3 = nn.Linear(4,1)
       
    def forward(self, x):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        out1 = self.C1(x)
        out1 = self.C11(out1)
        out1 = self.B1(out1)
        out1 = self.R1(out1)
        out1 = self.drop1(out1)

        out2 = self.C2(x)
        out2 = self.C21(out2)
        out2 = self.B2(out2)
        out2 = self.R2(out2)
        out2 = self.drop2(out2)

        out3 = self.C3(x)
        out3 = self.C31(out3)
        out3 = self.B3(out3)
        out3 = self.R3(out3)
        out3 = self.drop3(out3)  

        out4 = self.C4(x)
        out4 = self.C41(out4)
        out4 = self.B4(out4)
        out4 = self.R4(out4)
        out4 = self.drop4(out4)  

        out = torch.cat([out1,out2,out3,out4],1)
        #[batch_num, dim, len] ---> [batch_num,len,dim]
        out = self.encoder(out.transpose(2,1))
        #[batch_num,len,dim] ---> [batch_num,1,len]
        out = (self.F3(self.F2(self.F1(out)))).transpose(1,2)
        return out

class Transformer4(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.C1 = nn.Conv1d(1, 4, 256, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(4)
        self.R1 = nn.ReLU()
        self.C11 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B11 = nn.BatchNorm1d(32)
        self.R11 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.C2 = nn.Conv1d(1, 4, 128, stride=1, padding='same')
        self.B2 = nn.BatchNorm1d(4)
        self.R2 = nn.ReLU()
        self.C21 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B21 = nn.BatchNorm1d(32)
        self.R21 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.C3 = nn.Conv1d(1, 4, 64, stride=1, padding='same')
        self.B3 = nn.BatchNorm1d(4)
        self.R3 = nn.ReLU()
        self.C31 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B31 = nn.BatchNorm1d(32)
        self.R31 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.C4 = nn.Conv1d(1, 4, 32, stride=1, padding='same')
        self.B4 = nn.BatchNorm1d(4)
        self.R4 = nn.ReLU()
        self.C41 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B41 = nn.BatchNorm1d(32)
        self.R41 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)

        self.C5 = nn.Conv1d(1, 4, 16, stride=1, padding='same')
        self.B5 = nn.BatchNorm1d(4)
        self.R5 = nn.ReLU()
        self.C51 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B51 = nn.BatchNorm1d(32)
        self.R51 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.3)

        self.C6 = nn.Conv1d(1, 4, 8, stride=1, padding='same')
        self.B6 = nn.BatchNorm1d(4)
        self.R6 = nn.ReLU()
        self.C61 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B61 = nn.BatchNorm1d(32)
        self.R61 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.3)

        self.C7 = nn.Conv1d(1, 4, 4, stride=1, padding='same')
        self.B7 = nn.BatchNorm1d(4)
        self.R7 = nn.ReLU()
        self.C71 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B71 = nn.BatchNorm1d(32)
        self.R71 = nn.ReLU()
        self.drop7 = nn.Dropout(p=0.3)

        self.C8 = nn.Conv1d(1, 4, 2, stride=1, padding='same')
        self.B8 = nn.BatchNorm1d(4)
        self.R8 = nn.ReLU()
        self.C81 = nn.Conv1d(4, 32, 35, stride=1, padding='same')
        self.B81 = nn.BatchNorm1d(32)
        self.R81 = nn.ReLU()
        self.drop8 = nn.Dropout(p=0.3)

        self.encodelayer = nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True,dropout=0.3)
        self.encoder = nn.TransformerEncoder(self.encodelayer, num_layers=3)

        self.F1 = nn.Linear(256,32)
        self.F2 = nn.Linear(32,8)
        self.F3 = nn.Linear(8,1)
       
    def forward(self, x):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        out1 = self.C1(x)
        out1 = self.B1(out1)
        out1 = self.R1(out1)
        out1 = self.C11(out1)
        out1 = self.B11(out1)
        out1 = self.R11(out1)
        out1 = self.drop1(out1)

        out2 = self.C2(x)
        out2 = self.B2(out2)
        out2 = self.R2(out2)
        out2 = self.C21(out2)
        out2 = self.B21(out2)
        out2 = self.R21(out2)
        out2 = self.drop2(out2)

        out3 = self.C3(x)
        out3 = self.B3(out3)
        out3 = self.R3(out3)
        out3 = self.C31(out3)
        out3 = self.B31(out3)
        out3 = self.R31(out3)
        out3 = self.drop3(out3)  

        out4 = self.C4(x)
        out4 = self.B4(out4)
        out4 = self.R4(out4)
        out4 = self.C41(out4)
        out4 = self.B41(out4)
        out4 = self.R41(out4)
        out4 = self.drop4(out4)  

        out5 = self.C5(x)
        out5 = self.B5(out5)
        out5 = self.R5(out5)
        out5 = self.C51(out5)
        out5 = self.B51(out5)
        out5 = self.R51(out5)
        out5 = self.drop5(out5)  

        out6 = self.C6(x)
        out6 = self.B6(out6)
        out6 = self.R6(out6)
        out6 = self.C61(out6)
        out6 = self.B61(out6)
        out4 = self.R61(out6)
        out6 = self.drop6(out6)  

        out7 = self.C7(x)
        out7 = self.B7(out7)
        out7 = self.R7(out7)
        out7 = self.C71(out7)
        out7 = self.B71(out7)
        out7 = self.R71(out7)
        out7 = self.drop7(out7)  

        out8 = self.C8(x)
        out8 = self.B8(out8)
        out8 = self.R8(out8)
        out8 = self.C81(out8)
        out8 = self.B81(out8)
        out8 = self.R81(out8)
        out8 = self.drop8(out8)  

        out = torch.cat([out1,out2,out3,out4,out5,out6,out7,out8],1)
        print('形状个',out.shape)
        #[batch_num, dim, len] ---> [batch_num,len,dim]
        out = self.encoder(out.transpose(2,1))
        #[batch_num,len,dim] ---> [batch_num,1,len]
        out = (self.F3(self.F2(self.F1(out)))).transpose(1,2)
        return out

class Transformer5(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        #self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True,dropout=0.3), num_layers=2)
        nhead = 1
        kernal_num = 4
        self.C1 = nn.Conv1d(1, kernal_num, 256, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(kernal_num)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.encodelayer1 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E1 = nn.TransformerEncoder(self.encodelayer1, num_layers=2)


        self.C2 = nn.Conv1d(1, kernal_num, 128, stride=1, padding='same')
        self.B2 = nn.BatchNorm1d(kernal_num)
        self.R2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)
        self.encodelayer2 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E2 = nn.TransformerEncoder(self.encodelayer2, num_layers=2)

        self.C3 = nn.Conv1d(1, kernal_num, 64, stride=1, padding='same')
        self.B3 = nn.BatchNorm1d(kernal_num)
        self.R3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)
        self.encodelayer3 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E3 = nn.TransformerEncoder(self.encodelayer3, num_layers=2)

        self.C4 = nn.Conv1d(1, kernal_num, 32, stride=1, padding='same')
        self.B4 = nn.BatchNorm1d(kernal_num)
        self.R4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)
        self.encodelayer4 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E4 = nn.TransformerEncoder(self.encodelayer4, num_layers=2)

        self.C5 = nn.Conv1d(1, kernal_num, 16, stride=1, padding='same')
        self.B5 = nn.BatchNorm1d(kernal_num)
        self.R5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.3)
        self.encodelayer5 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E5 = nn.TransformerEncoder(self.encodelayer5, num_layers=2)

        self.C6 = nn.Conv1d(1, kernal_num, 8, stride=1, padding='same')
        self.B6 = nn.BatchNorm1d(kernal_num)
        self.R6 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.3)
        self.encodelayer6 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E6 = nn.TransformerEncoder(self.encodelayer6, num_layers=2)

        self.C7 = nn.Conv1d(1, kernal_num, 4, stride=1, padding='same')
        self.B7 = nn.BatchNorm1d(kernal_num)
        self.R7 = nn.ReLU()
        self.drop7 = nn.Dropout(p=0.3)
        self.encodelayer7 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E7 = nn.TransformerEncoder(self.encodelayer7, num_layers=2)

        self.C8 = nn.Conv1d(1, kernal_num, 2, stride=1, padding='same')
        self.B8 = nn.BatchNorm1d(kernal_num)
        self.R8 = nn.ReLU()
        self.drop8 = nn.Dropout(p=0.3)
        self.encodelayer8 = nn.TransformerEncoderLayer(d_model=4, nhead=nhead,batch_first=True,dropout=0.3)
        self.E8 = nn.TransformerEncoder(self.encodelayer8, num_layers=2)

        self.F1 = nn.Linear(32,8)
        self.F2 = nn.Linear(8,1)
       
    def forward(self, x):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        out1 = self.C1(x)
        out1 = self.B1(out1)
        out1 = self.R1(out1)
        out1 = self.drop1(out1)
        out1 = self.E1(out1.transpose(2,1))

        out2 = self.C2(x)
        out2 = self.B2(out2)
        out2 = self.R2(out2)
        out2 = self.drop2(out2)
        out2 = self.E2(out2.transpose(2,1))

        out3 = self.C3(x)
        out3 = self.B3(out3)
        out3 = self.R3(out3)
        out3 = self.drop3(out3)
        out3 = self.E3(out3.transpose(2,1)) 

        out4 = self.C4(x)
        out4 = self.B4(out4)
        out4 = self.R4(out4)
        out4 = self.drop4(out4)
        out4 = self.E4(out4.transpose(2,1))

        out5 = self.C5(x)
        out5 = self.B5(out5)
        out5 = self.R5(out5)
        out5 = self.drop5(out5)
        out5 = self.E5(out5.transpose(2,1)) 

        out6 = self.C6(x)
        out6 = self.B6(out6)
        out6 = self.R6(out6)
        out6 = self.drop6(out6)
        out6 = self.E6(out6.transpose(2,1))

        out7 = self.C7(x)
        out7 = self.B7(out7)
        out7 = self.R7(out7)
        out7 = self.drop7(out7)
        out7 = self.E7(out7.transpose(2,1)) 

        out8 = self.C8(x)
        out8 = self.B8(out8)
        out8 = self.R8(out8)
        out8 = self.drop8(out8)
        out8 = self.E8(out8.transpose(2,1))  

        out = torch.cat([out1,out2,out3,out4,out5,out6,out7,out8],2)
        out = (self.F2(self.F1(out))).transpose(1,2)
        return out

class Transformer5_1(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        #self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True,dropout=0.3), num_layers=2)
        nhead = 2
        kernal_num = 32
        self.C1 = nn.Conv1d(1, kernal_num, 256, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(kernal_num)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.encodelayer1 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E1 = nn.TransformerEncoder(self.encodelayer1, num_layers=1)


        self.C2 = nn.Conv1d(1, kernal_num, 128, stride=1, padding='same')
        self.B2 = nn.BatchNorm1d(kernal_num)
        self.R2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)
        self.encodelayer2 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E2 = nn.TransformerEncoder(self.encodelayer2, num_layers=1)

        self.C3 = nn.Conv1d(1, kernal_num, 64, stride=1, padding='same')
        self.B3 = nn.BatchNorm1d(kernal_num)
        self.R3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)
        self.encodelayer3 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E3 = nn.TransformerEncoder(self.encodelayer3, num_layers=1)

        self.C4 = nn.Conv1d(1, kernal_num, 32, stride=1, padding='same')
        self.B4 = nn.BatchNorm1d(kernal_num)
        self.R4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)
        self.encodelayer4 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E4 = nn.TransformerEncoder(self.encodelayer4, num_layers=1)

        self.C5 = nn.Conv1d(1, kernal_num, 16, stride=1, padding='same')
        self.B5 = nn.BatchNorm1d(kernal_num)
        self.R5 = nn.ReLU()
        self.drop5 = nn.Dropout(p=0.3)
        self.encodelayer5 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E5 = nn.TransformerEncoder(self.encodelayer5, num_layers=1)

        self.C6 = nn.Conv1d(1, kernal_num, 8, stride=1, padding='same')
        self.B6 = nn.BatchNorm1d(kernal_num)
        self.R6 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.3)
        self.encodelayer6 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E6 = nn.TransformerEncoder(self.encodelayer6, num_layers=1)

        self.C7 = nn.Conv1d(1, kernal_num, 4, stride=1, padding='same')
        self.B7 = nn.BatchNorm1d(kernal_num)
        self.R7 = nn.ReLU()
        self.drop7 = nn.Dropout(p=0.3)
        self.encodelayer7 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E7 = nn.TransformerEncoder(self.encodelayer7, num_layers=1)

        self.C8 = nn.Conv1d(1, kernal_num, 2, stride=1, padding='same')
        self.B8 = nn.BatchNorm1d(kernal_num)
        self.R8 = nn.ReLU()
        self.drop8 = nn.Dropout(p=0.3)
        self.encodelayer8 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=nhead,batch_first=True,dropout=0.3)
        self.E8 = nn.TransformerEncoder(self.encodelayer8, num_layers=1)


        self.encodelayer9 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=16,batch_first=True,dropout=0.5)
        self.E9 = nn.TransformerEncoder(self.encodelayer9, num_layers=3)

        self.F1 = nn.Linear(kernal_num,8)
        self.F2 = nn.Linear(8,1)

    def forward(self, x):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        out1 = self.C1(x)
        out1 = self.B1(out1)
        out1 = self.R1(out1)
        out11 = self.drop1(out1)
        out1 = self.E1(out11.transpose(2,1))

        out2 = self.C2(x)
        out2 = self.B2(out2)
        out2 = self.R2(out2)
        out21 = self.drop2(out2)
        out2 = self.E2(out21.transpose(2,1))

        out3 = self.C3(x)
        out3 = self.B3(out3)
        out3 = self.R3(out3)
        out31 = self.drop3(out3)
        out3 = self.E3(out31.transpose(2,1)) 

        out4 = self.C4(x)
        out4 = self.B4(out4)
        out4 = self.R4(out4)
        out41 = self.drop4(out4)
        out4 = self.E4(out41.transpose(2,1))

        out5 = self.C5(x)
        out5 = self.B5(out5)
        out5 = self.R5(out5)
        out51 = self.drop5(out5)
        out5 = self.E5(out51.transpose(2,1)) 

        out6 = self.C6(x)
        out6 = self.B6(out6)
        out6 = self.R6(out6)
        out61 = self.drop6(out6)
        out6 = self.E6(out61.transpose(2,1))

        out7 = self.C7(x)
        out7 = self.B7(out7)
        out7 = self.R7(out7)
        out71 = self.drop7(out7)
        out7 = self.E7(out71.transpose(2,1)) 

        out8 = self.C8(x)
        out8 = self.B8(out8)
        out8 = self.R8(out8)
        out81 = self.drop8(out8)
        out8 = self.E8(out81.transpose(2,1))  

        out = out1+out2+out3+out4+out5+out6+out7+out8+out81.transpose(2,1)+out11.transpose(2,1)+out21.transpose(2,1)+out31.transpose(2,1)+out41.transpose(2,1)+out51.transpose(2,1)+out61.transpose(2,1)+out71.transpose(2,1)
        out = self.E9(out)
        out = (self.F2(self.F1(out))).transpose(1,2)
        return out

class attconv_block(nn.Module):
    def __init__(self,c, k, T):
        super(attconv_block, self).__init__() 
        self.covn1d_1 = nn.Conv1d(in_channels = 1, out_channels = c, kernel_size = k,padding = 'same')
        self.drop1 = nn.Dropout(p=0.7)
        self.covn1d_2 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = 'same')
        self.drop2 = nn.Dropout(p=0.7)
        self.covn1d_3 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = 'same')
        self.drop3 = nn.Dropout(p=0.7)
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=c, nhead=4,batch_first=True,dropout=0.7), num_layers=1)
        
    def forward(self, x):
        x = self.drop1(torch.relu(self.covn1d_1(x)))
        x = self.drop2(torch.relu(self.covn1d_2(x) + x)) 
        x = self.drop3(torch.relu(self.covn1d_3(x) + x))
        x = torch.relu(self.attention(x.transpose(2,1)).transpose(2,1) + x)
        return x

class Transformer5_2(nn.Module):
    def __init__(self,c,datanum) -> None:
        super().__init__()
        #nhead = 2
        #kernal_num = 32
        self.block_1 = attconv_block(c=32, k=256, T=datanum)
        self.block_2 = attconv_block(c=32, k=128, T=datanum)
        self.block_3 = attconv_block(c=32, k=64, T=datanum)
        self.block_4 = attconv_block(c=32, k=32, T=datanum)
        self.block_5 = attconv_block(c=32, k=16, T=datanum)
        self.block_6 = attconv_block(c=32, k=8, T=datanum)
        self.block_7 = attconv_block(c=32, k=4, T=datanum)
        self.block_8 = attconv_block(c=32, k=2, T=datanum)
        self.conv1d1 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 25,padding = 'same')
        self.dropout1 = nn.Dropout(p=0.7)
        self.conv1d2 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 25,padding = 'same')
        self.dropout2 = nn.Dropout(p=0.7)
        self.conv1d3 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 25,padding = 'same')
        self.dropout3 = nn.Dropout(p=0.7)
        #self.encodelayer9 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=16,batch_first=True,dropout=0.5)
        #self.E9 = nn.TransformerEncoder(self.encodelayer9, num_layers=3)
        self.F = nn.Linear(c * datanum, datanum)

    def forward(self, x):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
        x = self.dropout1(torch.relu(self.conv1d1(x)))
        x = self.dropout2(torch.relu(self.conv1d2(x) + x)) 
        x = self.dropout3(torch.relu(self.conv1d3(x) + x))
        out =  self.F(x.view(x.size(0),-1)).unsqueeze(1)
        return out


class conv_block(nn.Module):
    def __init__(self,k, k1=24):
        super(conv_block, self).__init__() 
        self.cc1 = nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size =k,stride = 2,padding=int((k-2)/2))
        self.drop1 = nn.Dropout(p=0.7)
        self.cc2 = nn.Conv1d(4,16,kernel_size = k1,stride = 2,padding=int((k1-2)/2))
        self.drop2 = nn.Dropout(p=0.7)
        self.cc3 = nn.Conv1d(16,32,kernel_size = k1,stride = 2,padding=int((k1-2)/2)) 
        self.drop3 = nn.Dropout(p=0.7)   
        #self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=c, nhead=4,batch_first=True,dropout=0.7), num_layers=1)
        
    def forward(self, x):
        x = self.drop1(torch.relu(self.cc1(x)))
        x = self.drop2(torch.relu(self.cc2(x))) 
        x = self.drop3(torch.relu(self.cc3(x)))
        #x = torch.relu(self.attention(x.transpose(2,1)).transpose(2,1) + x)
        return x
class attn_block(nn.Module):
    def __init__(self,datanum):
        super(attn_block, self).__init__() 
        self.attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True,dropout=0.7), num_layers=4)
        self.F1 = nn.Linear(256*64,datanum*4)
        self.drop = nn.Dropout(p=0.7)
        self.F2 = nn.Linear(datanum*4,datanum)
    def forward(self,x):
        x = torch.relu(self.drop(self.attention(x.transpose(0,2).transpose(1,2))))
        x = x.transpose(1,2).transpose(0,2)
        #print(x.shape)
        x = torch.flatten(x,1,2)
        x = torch.relu(self.drop(self.F1(x)))
        x = self.F2(x)
        return x
class Transformer5_3(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.attn = attn_block(datanum)
        
    def forward(self, x):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = self.attn(x)
        
        out =  x.unsqueeze(1)
        return out

class probattn_block(nn.Module):
    def __init__(self,datanum):
        super(probattn_block, self).__init__() 
        self.datanum = datanum
        self.drop = nn.Dropout(p=0.7)
        self.attention = AttentionLayer(ProbAttention(attention_dropout=0.7),d_model=32,n_heads=8)
        self.BN1 = nn.BatchNorm1d(32)
        self.F1 = nn.Linear(self.datanum*4,self.datanum*8)
        self.F2 = nn.Linear(self.datanum*8,self.datanum*4)
        self.BN2 = nn.BatchNorm1d(32)

    def forward(self,x):
        res1 = x
        x = x.transpose(0,2).transpose(1,2)
        x,_ = self.attention(x,x,x,attn_mask=None)
        x = torch.relu(self.drop(x))
        x = self.BN1(x.transpose(0,1).transpose(1,2)+res1)
        res2 = x
        x = torch.flatten(x,1,2)
        x = torch.relu(self.drop(self.F1(x)))
        x = torch.relu(self.drop(self.F2(x)))
        x = x.view(x.size(0),32,int(self.datanum/8))
        x = self.BN2(x+res2)

        return x
class Informer5_3(nn.Module):
    def __init__(self,datanum):
        super(Informer5_3, self).__init__()    

        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.probattn1 = probattn_block(datanum)
        self.probattn2 = probattn_block(datanum)
        self.probattn3 = probattn_block(datanum)
        self.F = nn.Linear(datanum*4,datanum)
    def forward(self, x,attn_mask=None):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        #x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = x1+x2+x3+x4+x5+x6+x7+x8
        x = self.probattn3(self.probattn2(self.probattn1(x)))
        x = torch.flatten(x,1,2)
        x = self.F(x)
        out =  x.unsqueeze(1)
        return out


class Informer5_4(nn.Module):
    def __init__(self,datanum):
        super(Informer5_4, self).__init__()    

        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.probattn1 = probattn_block(datanum)
        self.probattn2 = probattn_block(datanum)
        self.probattn3 = probattn_block(datanum)
        self.probattn4 = probattn_block(datanum)
        self.F = nn.Linear(datanum*4,datanum)
    def forward(self, x,attn_mask=None):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        #x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = x1+x2+x3+x4+x5+x6+x7+x8
        x = self.probattn4(self.probattn3(self.probattn2(self.probattn1(x))))
        x = torch.flatten(x,1,2)
        x = self.F(x)
        out =  x.unsqueeze(1)
        return out


        

class Informer5_5(nn.Module):
    def __init__(self,datanum):
        super(Informer5_5, self).__init__()    
        self.datanum = datanum
        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.probattn1 = probattn_block(self.datanum)
        self.probattn2 = probattn_block(self.datanum)
        self.probattn3 = probattn_block(self.datanum)
        self.probattn4 = probattn_block(self.datanum)
        self.probattn5 = probattn_block(self.datanum)
        self.F = nn.Linear(self.datanum*4,self.datanum)

    def forward(self, x, test = False, attn_mask=None):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        #x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = x1+x2+x3+x4+x5+x6+x7+x8
        x = self.probattn5(self.probattn4(self.probattn3(self.probattn2(self.probattn1(x)))))
        x = torch.flatten(x,1,2)
        x = self.F(x)
        x = x.unsqueeze(1)

        return x

class Informer5_6(nn.Module):
    def __init__(self,datanum):
        super(Informer5_6, self).__init__()    
        self.datanum = datanum
        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.probattn1 = probattn_block(self.datanum)
        self.probattn2 = probattn_block(self.datanum)
        self.probattn3 = probattn_block(self.datanum)
        self.probattn4 = probattn_block(self.datanum)
        self.probattn5 = probattn_block(self.datanum)
        self.probattn6 = probattn_block(self.datanum)
        self.F = nn.Linear(self.datanum*4,self.datanum)

    def forward(self, x, test = False, attn_mask=None):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        #x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = x1+x2+x3+x4+x5+x6+x7+x8
        x = self.probattn6(self.probattn5(self.probattn4(self.probattn3(self.probattn2(self.probattn1(x))))))
        x = torch.flatten(x,1,2)
        x = self.F(x)
        x = x.unsqueeze(1)

        return x

    
class WaveInformer5_5(nn.Module):
    def __init__(self,datanum):
        super(WaveInformer5_5, self).__init__()    
        self.datanum = datanum
        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.probattn1 = probattn_block(self.datanum)
        self.probattn2 = probattn_block(self.datanum)
        self.probattn3 = probattn_block(self.datanum)
        self.probattn4 = probattn_block(self.datanum)
        self.probattn5 = probattn_block(self.datanum)
        self.F = nn.Linear(self.datanum*4,self.datanum)
        self.dwt = DWT1DForward(wave='db2', J=6)
        self.idwt = DWT1DInverse(wave='db2')
    def forward(self, x, test = False, attn_mask=None):
        xl,xh = self.dwt(x)
        x = self.idwt((xl,xh))
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        #x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = x1+x2+x3+x4+x5+x6+x7+x8
        x = self.probattn5(self.probattn4(self.probattn3(self.probattn2(self.probattn1(x)))))
        x = torch.flatten(x,1,2)
        x = self.F(x)
        x = x.unsqueeze(1)


        return x
    

class WaveInformer5_5hou(nn.Module):
    def __init__(self,datanum):
        super(WaveInformer5_5hou, self).__init__()    
        self.datanum = datanum
        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.probattn1 = probattn_block(self.datanum)
        self.probattn2 = probattn_block(self.datanum)
        self.probattn3 = probattn_block(self.datanum)
        self.probattn4 = probattn_block(self.datanum)
        self.probattn5 = probattn_block(self.datanum)
        self.F = nn.Linear(self.datanum*4,self.datanum)
        self.dwt = DWT1DForward(wave='db2', J=6)
        self.idwt = DWT1DInverse(wave='db2')
    def forward(self, x, test = False, attn_mask=None):
    
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        #x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = x1+x2+x3+x4+x5+x6+x7+x8
        x = self.probattn5(self.probattn4(self.probattn3(self.probattn2(self.probattn1(x)))))
        x = torch.flatten(x,1,2)
        x = self.F(x)
        x = x.unsqueeze(1)
        xl,xh = self.dwt(x)
        x = self.idwt((xl,xh))
        


        return x



class Informer5_8(nn.Module):
    def __init__(self,datanum):
        super(Informer5_8, self).__init__()    

        #nhead = 2
        #kernal_num = 32
        self.block_1 = conv_block(k=256)
        self.block_2 = conv_block(k=128)
        self.block_3 = conv_block(k=64)
        self.block_4 = conv_block(k=32)
        self.block_5 = conv_block(k=16)
        self.block_6 = conv_block(k=8)
        self.block_7 = conv_block(k=4)
        self.block_8 = conv_block(k=2)
        self.probattn1 = probattn_block(datanum)
        self.probattn2 = probattn_block(datanum)
        self.probattn3 = probattn_block(datanum)
        self.probattn4 = probattn_block(datanum)
        self.probattn5 = probattn_block(datanum)
        self.probattn6 = probattn_block(datanum)
        self.probattn7 = probattn_block(datanum)
        self.probattn8 = probattn_block(datanum)
        self.F = nn.Linear(datanum*4,datanum)
    def forward(self, x,attn_mask=None):
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        #x = torch.cat([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8],axis=1)
        x = x1+x2+x3+x4+x5+x6+x7+x8
        x = self.probattn8(self.probattn7(self.probattn6(self.probattn5(self.probattn4(self.probattn3(self.probattn2(self.probattn1(x))))))))
        x = torch.flatten(x,1,2)
        x = self.F(x)
        out =  x.unsqueeze(1)
        return out



class EMG_attconv_block(nn.Module):
    def __init__(self,c, k, T):
        super(EMG_attconv_block, self).__init__() 
        self.covn1d_1 = nn.Conv1d(in_channels = 1, out_channels = c, kernel_size = k,padding = 'same')
        self.drop1 = nn.Dropout(p=0.7)
        self.covn1d_2 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = 'same')
        self.drop2 = nn.Dropout(p=0.7)
        self.covn1d_3 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = 'same')
        self.drop3 = nn.Dropout(p=0.7)
        self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=c, nhead=4,batch_first=True,dropout=0.7), num_layers=1)
        #self.fc_1 = nn.Linear(c * T, T)
        #self.fc_2 = nn.Linear(c * T, T)
        
    def forward(self, x):
        x = self.drop1(torch.relu(self.covn1d_1(x)))
        x = self.drop2(torch.relu(self.covn1d_2(x) + x)) 
        x = self.drop3(torch.relu(self.covn1d_3(x) + x))
        x = torch.relu(self.attention(x.transpose(2,1)).transpose(2,1) + x)
        return x

class EMG_Transformer5_2(nn.Module):
    def __init__(self,c,datanum) -> None:
        super().__init__()
        #self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True,dropout=0.3), num_layers=2)
        nhead = 2
        kernal_num = 32
        self.block_1 = attconv_block(c=c, k=256, T=datanum)
        self.block_2 = attconv_block(c=c, k=128, T=datanum)
        self.block_3 = attconv_block(c=c, k=64, T=datanum)
        self.block_4 = attconv_block(c=c, k=32, T=datanum)
        self.block_5 = attconv_block(c=c, k=16, T=datanum)
        self.block_6 = attconv_block(c=c, k=8, T=datanum)
        self.block_7 = attconv_block(c=c, k=4, T=datanum)
        self.block_8 = attconv_block(c=c, k=512, T=datanum)

        self.conv1d1 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 103,padding = 'same')
        self.dropout1 = nn.Dropout(p=0.7)
        self.conv1d2 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 103,padding = 'same')
        self.dropout2 = nn.Dropout(p=0.7)
        self.conv1d3 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 103,padding = 'same')
        self.dropout3 = nn.Dropout(p=0.7)
        #self.encodelayer9 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=16,batch_first=True,dropout=0.5)
        #self.E9 = nn.TransformerEncoder(self.encodelayer9, num_layers=3)

        self.F = nn.Linear(c * datanum, datanum)
        
    def forward(self, x):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

        x = self.dropout1(torch.relu(self.conv1d1(x)))
        x = self.dropout2(torch.relu(self.conv1d2(x) + x)) 
        x = self.dropout3(torch.relu(self.conv1d3(x) + x))
        
        out =  self.F(x.view(x.size(0),-1)).unsqueeze(1)
        return out

'''class Transformer5_3(nn.Module):
    def __init__(self,c,datanum) -> None:
        super().__init__()
        #self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True,dropout=0.3), num_layers=2)
        nhead = 2
        kernal_num = 32
        self.block_1 = attconv_block(c=32, k=256, T=datanum)
        self.block_2 = attconv_block(c=32, k=128, T=datanum)
        self.block_3 = attconv_block(c=32, k=64, T=datanum)
        self.block_4 = attconv_block(c=32, k=32, T=datanum)
        self.block_5 = attconv_block(c=32, k=16, T=datanum)
        self.block_6 = attconv_block(c=32, k=8, T=datanum)
        self.block_7 = attconv_block(c=32, k=4, T=datanum)
        self.block_8 = attconv_block(c=32, k=2, T=datanum)

        self.conv1d1 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 35,padding = 'same')
        self.dropout1 = nn.Dropout(p=0.3)
        self.conv1d2 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 35,padding = 'same')
        self.dropout2 = nn.Dropout(p=0.3)
        self.conv1d3 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = 35,padding = 'same')
        self.dropout3 = nn.Dropout(p=0.3)
        self.atte = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=c, nhead=16,batch_first=True), num_layers=1)
        self.dropout4 = nn.Dropout(p=0.3)
        #self.encodelayer9 = nn.TransformerEncoderLayer(d_model=kernal_num, nhead=16,batch_first=True,dropout=0.5)
        #self.E9 = nn.TransformerEncoder(self.encodelayer9, num_layers=3)

        self.F = nn.Linear(c * datanum, datanum)

    def forward(self, x):
        # [batch_num, 1, len] ---> [batch_num, dim, len]
        x1 = self.block_1(x)
        x2 = self.block_2(x)
        x3 = self.block_3(x)
        x4 = self.block_4(x)
        x5 = self.block_5(x)
        x6 = self.block_6(x)
        x7 = self.block_7(x)
        x8 = self.block_8(x)
        x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

        x = self.dropout1(torch.relu(self.conv1d1(x)))
        x = self.dropout2(torch.relu(self.conv1d2(x) + x)) 
        x = self.dropout3(torch.relu(self.conv1d3(x) + x))
        x = self.dropout4(torch.relu(self.atte(x.transpose(2,1)).transpose(2,1) + x))
        out =  self.F(x.contiguous().view(x.size(0),-1)).unsqueeze(1)
        return out

'''
class fcNN_attn(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.f1 = nn.Linear(datanum, datanum)
        self.drop1 = nn.Dropout(p=0.3)

        self.f2 = nn.Linear(datanum, datanum)
        self.a1 = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=1)
        self.r1 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.f3 = nn.Linear(datanum, datanum)
        self.r2 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.f4 = nn.Linear(datanum, datanum)

    def forward(self, x):
        out = self.f1(x)
        out = self.drop1(out)

        out = self.f2(out)
        out = self.a1(out,out,out)
        out = self.r1(out)
        out = self.drop2(out)

        out = self.f3(out)
        out = self.r2(out)
        out = self.drop3(out)

        out = self.f4(out)
        return out

class fcNN(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.f1 = nn.Linear(datanum, datanum)
        self.drop1 = nn.Dropout(p=0.3)

        self.f2 = nn.Linear(datanum, datanum)
        self.r1 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.f3 = nn.Linear(datanum, datanum)
        self.r2 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.f4 = nn.Linear(datanum, datanum)

    def forward(self, x):
        out = self.f1(x)
        out = self.drop1(out)

        out = self.f2(out)
        out = self.r1(out)
        out = self.drop2(out)

        out = self.f3(out)
        out = self.r2(out)
        out = self.drop3(out)

        out = self.f4(out)
        return out

class simple_CNN(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.C1 = nn.Conv1d(1, 64, 3, stride=1, padding='same')
        self.B1 = nn.BatchNorm1d(64)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.C2 = nn.Conv1d(64, 64, 3, stride=1, padding='same')
        self.B2 = nn.BatchNorm1d(64)
        self.R2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.C3 = nn.Conv1d(64, 64, 3, stride=1, padding='same')
        self.B3 = nn.BatchNorm1d(64)
        self.R3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.C4 = nn.Conv1d(64, 64, 3, stride=1, padding='same')
        self.B4 = nn.BatchNorm1d(64)
        self.R4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)

        self.f = nn.Flatten()
        self.F = nn.Linear(datanum * 64, datanum)

    def forward(self, x):
        out = self.C1(x)
        out = self.B1(out)
        out = self.R1(out)
        out = self.drop1(out)

        out = self.C2(out)
        out = self.B2(out)
        out = self.R2(out)
        out = self.drop2(out)

        out = self.C3(out)
        out = self.B3(out)
        out = self.R3(out)
        out = self.drop3(out)
        
        out = self.C4(out)
        out = self.B4(out)
        out = self.R4(out)
        out = self.drop4(out)
        #print(out.shape,type(out))
        out = self.f(out)
        out = (self.f(out)).unsqueeze(1)
        out = self.F(out)
        return out

class simple_CNN_attn1(nn.Module):
    def __init__(self,datanum) -> None:
        super().__init__()
        self.C1 = nn.Conv1d(1, 64, 3, stride=1, padding='same')
        self.a1 = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=4)
        self.B1 = nn.BatchNorm1d(64)
        self.R1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)

        self.C2 = nn.Conv1d(64, 64, 3, stride=1, padding='same')
        self.a2 = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=4)
        self.B2 = nn.BatchNorm1d(64)
        self.R2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)

        self.C3 = nn.Conv1d(64, 64, 3, stride=1, padding='same')
        self.a3 = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=4)
        self.B3 = nn.BatchNorm1d(64)
        self.R3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)

        self.C4 = nn.Conv1d(64, 64, 3, stride=1, padding='same')
        self.a4 = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=4)
        self.B4 = nn.BatchNorm1d(64)
        self.R4 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.3)

        self.f = nn.Flatten()
        self.F = nn.Linear(datanum * 64, datanum)

    def forward(self, x):
        out = self.C1(x)
        out = self.a1(out,out,out)
        out = self.B1(out)
        out = self.R1(out)
        out = self.drop1(out)

        out = self.C2(out)
        #out = self.a2(out,out,out)
        out = self.B2(out)
        out = self.R2(out)
        out = self.drop2(out)

        out = self.C3(out)
        #out = self.a3(out,out,out)
        out = self.B3(out)
        out = self.R3(out)
        out = self.drop3(out)
        
        out = self.C4(out)
        out = self.a4(out,out,out)
        out = self.B4(out)
        out = self.R4(out)
        out = self.drop4(out)
      
        out = (self.f(out)).unsqueeze(1)
        out = self.F(out)
        
        return out

class block(nn.Module):
    def __init__(self,c, k, T):
        super(block, self).__init__() 
        self.covn1d_1 = nn.Conv1d(in_channels = 1, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.covn1d_2 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.covn1d_3 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.covn1d_4 = nn.Conv1d(in_channels = c, out_channels = c, kernel_size = k,padding = int((k-1)/2))
        self.fc_1 = nn.Linear(c * T, T)
        self.fc_2 = nn.Linear(c * T, T)
        
    def forward(self, x):
        x = torch.relu(self.covn1d_1(x))
        x = torch.relu(self.covn1d_2(x) + x)    
        x = torch.relu(self.covn1d_3(x) + x)
        x = torch.relu(self.covn1d_4(x) + x)
        signal =  self.fc_1(x.view(x.size(0),-1)).unsqueeze(1)
        noise = self.fc_2(x.view(x.size(0),-1)).unsqueeze(1)
        return signal, noise
class MMNN_4(nn.Module):
    def __init__(self,T,c=32, k=25):
        super(MMNN_4, self).__init__()
        self.block_1 = block(c, k, T)
        self.block_2 = block(c, k, T)
        self.block_3 = block(c, k, T)
        self.block_4 = block(c, k, T)
    def forward(self, x):
        signal_1, noise_1 = self.block_1(x)
        signal_2, noise_2 = self.block_2(x - noise_1)
        signal_3, noise_3 = self.block_3(x - noise_2)
        signal_4, noise_4 = self.block_4(x - noise_3)
        return signal_1 + signal_2 + signal_3 + signal_4


class LSTM(nn.Module):
    def __init__(self, datanum):
        super(LSTM, self).__init__()
        #batch, feature,seq,
        #seq, batch, feature
        self.LSTM = nn.LSTM(1,1,1)

        self.F1 = nn.Linear(datanum,datanum)
        self.Dropout1 = nn.Dropout(0.3)

        self.F2 = nn.Linear(datanum,datanum)
        self.Dropout2 = nn.Dropout(0.3)  

        self.F3 = nn.Linear(datanum,datanum)  

    def forward(self,x):
        x = x.transpose(2,0).transpose(1,2)
        out,_ = self.LSTM(x)
        out = out.transpose(1,2).transpose(0,2)
        out = self.Dropout1(torch.relu(self.F1(out)))
        out = self.Dropout2(torch.relu(self.F2(out)))
        
        out = self.F3(out)
        return out 


class Novel_CNN(nn.Module):
    def __init__(self,datanum):
        super(Novel_CNN, self).__init__()
        self.Conv1 = nn.Conv1d(1,32,3,1,padding='same')
        self.Conv2 = nn.Conv1d(32,32,3,1,padding='same')
        self.Pooling = nn.AvgPool1d(2,2)
        self.Conv3 = nn.Conv1d(32,64,3,1,padding='same')
        self.Conv4 = nn.Conv1d(64,64,3,1,padding='same')

        self.Conv5 = nn.Conv1d(64,128,3,1,padding='same')
        self.Conv6 = nn.Conv1d(128,128,3,1,padding='same')

        self.Conv7 = nn.Conv1d(128,256,3,1,padding='same')
        self.Conv8 = nn.Conv1d(256,256,3,1,padding='same')
        self.dropout = nn.Dropout(p=0.5)

        self.Conv9 = nn.Conv1d(256,512,3,1,padding='same')
        self.Conv10 = nn.Conv1d(512,512,3,1,padding='same')

        self.Conv11 = nn.Conv1d(512,1024,3,1,padding='same')
        self.Conv12 = nn.Conv1d(1024,1024,3,1,padding='same')

        self.Conv13 = nn.Conv1d(1024,2048,3,1,padding='same')
        self.Conv14 = nn.Conv1d(2048,2048,3,1,padding='same')

        self.F = nn.Linear(datanum*32,datanum)

    def forward(self,x):
        x = self.Pooling(torch.relu(self.Conv2(torch.relu(self.Conv1(x)))))
        x = self.Pooling(torch.relu(self.Conv4(torch.relu(self.Conv3(x)))))
        x = self.Pooling(torch.relu(self.Conv6(torch.relu(self.Conv5(x)))))
        x = self.Pooling(self.dropout(torch.relu(self.Conv8(torch.relu(self.Conv7(x))))))
        x = self.Pooling(self.dropout(torch.relu(self.Conv10(torch.relu(self.Conv9(x))))))
        x = self.Pooling(self.dropout(torch.relu(self.Conv12(torch.relu(self.Conv11(x))))))
        x = self.dropout(torch.relu(self.Conv14(torch.relu(self.Conv13(x)))))
        x = self.F(torch.flatten(x,1,2))
        x = x.unsqueeze(1)
        return x


# 主函数
if __name__ == '__main__':
    model = Informer5_5(512)

    # 测试网络正确性, 一般是给定一个确定的输入尺寸, 看输出的尺寸是不是我们想要的
    input = torch.ones((40,1,512))     # batch_size=64,代表64张图片  3个通道, CIFAR10 是32*32
    output = model(input)                   # 将输入送入网络中，得到输出output
    print('输出形状',output.shape)


# 当我们输入64张图片的时候，可以看到返回64行数据，每一行数据上有10个数据， 这10个数据代表每一张图片在我们10个类别中的概率
'''if __name__=="__main__":
    net = Complex_CNN(512)
    input = torch.ones((40,1,512))
    onnx = torch.onnx.export(net, input, 'Complex_CNN.onnx')
    netron.start(onnx)'''
