import torch
import torch.nn as nn


class DeepSeparator(nn.Module):

    def __init__(self):
        super(DeepSeparator, self).__init__()

        self.conv1_1_1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv1_1_2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2)
        self.conv1_1_3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, padding=5)
        self.conv1_1_4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=7)

        self.conv1_2_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv1_2_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv1_2_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv1_2_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv1_3_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv1_3_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv1_3_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv1_3_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv1_4_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv1_4_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv1_4_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv1_4_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv1_squeeze1 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)

        '''--------------------------------------------------------------------------------'''

        self.conv2_1_1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2_1_2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2)
        self.conv2_1_3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, padding=5)
        self.conv2_1_4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=7)

        self.conv2_2_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv2_2_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv2_2_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv2_2_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv2_3_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv2_3_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv2_3_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv2_3_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv2_4_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv2_4_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv2_4_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv2_4_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv1_squeeze2 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)

        '''--------------------------------------------------------------------------------'''

        self.conv3_1_1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv3_1_2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2)
        self.conv3_1_3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, padding=5)
        self.conv3_1_4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=7)

        self.conv3_2_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv3_2_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv3_2_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv3_2_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv3_3_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv3_3_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv3_3_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv3_3_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv3_4_1 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=3, padding=1)
        self.conv3_4_2 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=5, padding=2)
        self.conv3_4_3 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=11, padding=5)
        self.conv3_4_4 = nn.Conv1d(in_channels=20, out_channels=5, kernel_size=15, padding=7)

        self.conv1_squeeze3 = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)

        '''--------------------------------------------------------------------------------'''

        self.batch_norm = nn.BatchNorm1d(512, affine=True)

    def forward(self, x, indicator):

        emb_x = x
        learnable_atte_x = x

        emb_x = torch.unsqueeze(emb_x, 1)

        emb_x_1 = self.conv1_1_1(emb_x)
        emb_x_2 = self.conv1_1_2(emb_x)
        emb_x_3 = self.conv1_1_3(emb_x)
        emb_x_4 = self.conv1_1_4(emb_x)

        emb_x = torch.cat((emb_x_1, emb_x_2, emb_x_3, emb_x_4), dim=1)
        emb_x = torch.relu(emb_x)
        emb_skip_connect_x = emb_x

        emb_x_1 = self.conv1_2_1(emb_x)
        emb_x_2 = self.conv1_2_2(emb_x)
        emb_x_3 = self.conv1_2_3(emb_x)
        emb_x_4 = self.conv1_2_4(emb_x)
        emb_x = torch.cat((emb_x_1, emb_x_2, emb_x_3, emb_x_4), dim=1)
        emb_x = torch.sigmoid(emb_x)

        emb_x_1 = self.conv1_3_1(emb_x)
        emb_x_2 = self.conv1_3_2(emb_x)
        emb_x_3 = self.conv1_3_3(emb_x)
        emb_x_4 = self.conv1_3_4(emb_x)
        emb_x = torch.cat((emb_x_1, emb_x_2, emb_x_3, emb_x_4), dim=1)
        emb_x = torch.sigmoid(emb_x)
        emb_x = emb_x + emb_skip_connect_x

        emb_x_1 = self.conv1_4_1(emb_x)
        emb_x_2 = self.conv1_4_2(emb_x)
        emb_x_3 = self.conv1_4_3(emb_x)
        emb_x_4 = self.conv1_4_4(emb_x)
        emb_x = torch.cat((emb_x_1, emb_x_2, emb_x_3, emb_x_4), dim=1)
        emb_x = self.conv1_squeeze1(emb_x)

        emb_x = torch.squeeze(emb_x, 1)

        '''--------------------------------------------------------------------------------'''

        learnable_atte_x = torch.unsqueeze(learnable_atte_x, 1)

        learnable_atte_x_1 = self.conv2_1_1(learnable_atte_x)
        learnable_atte_x_2 = self.conv2_1_2(learnable_atte_x)
        learnable_atte_x_3 = self.conv2_1_3(learnable_atte_x)
        learnable_atte_x_4 = self.conv2_1_4(learnable_atte_x)
        learnable_atte_x = torch.cat((learnable_atte_x_1, learnable_atte_x_2, learnable_atte_x_3, learnable_atte_x_4),
                                     dim=1)
        learnable_atte_x = torch.relu(learnable_atte_x)
        atte_skip_connect_x = learnable_atte_x

        learnable_atte_x_1 = self.conv2_2_1(learnable_atte_x)
        learnable_atte_x_2 = self.conv2_2_2(learnable_atte_x)
        learnable_atte_x_3 = self.conv2_2_3(learnable_atte_x)
        learnable_atte_x_4 = self.conv2_2_4(learnable_atte_x)
        learnable_atte_x = torch.cat((learnable_atte_x_1, learnable_atte_x_2, learnable_atte_x_3, learnable_atte_x_4),
                                     dim=1)
        learnable_atte_x = torch.sigmoid(learnable_atte_x)

        learnable_atte_x_1 = self.conv2_3_1(learnable_atte_x)
        learnable_atte_x_2 = self.conv2_3_2(learnable_atte_x)
        learnable_atte_x_3 = self.conv2_3_3(learnable_atte_x)
        learnable_atte_x_4 = self.conv2_3_4(learnable_atte_x)
        learnable_atte_x = torch.cat((learnable_atte_x_1, learnable_atte_x_2, learnable_atte_x_3, learnable_atte_x_4),
                                     dim=1)
        learnable_atte_x = torch.sigmoid(learnable_atte_x)
        learnable_atte_x = learnable_atte_x + atte_skip_connect_x

        learnable_atte_x_1 = self.conv2_4_1(learnable_atte_x)
        learnable_atte_x_2 = self.conv2_4_2(learnable_atte_x)
        learnable_atte_x_3 = self.conv2_4_3(learnable_atte_x)
        learnable_atte_x_4 = self.conv2_4_4(learnable_atte_x)
        learnable_atte_x = torch.cat((learnable_atte_x_1, learnable_atte_x_2, learnable_atte_x_3, learnable_atte_x_4),
                                     dim=1)

        learnable_atte_x = self.conv1_squeeze2(learnable_atte_x)
        learnable_atte_x = torch.sigmoid(learnable_atte_x)
        learnable_atte_x = torch.squeeze(learnable_atte_x, 1)

        '''--------------------------------------------------------------------------------'''

        atte_x = indicator - learnable_atte_x
        atte_x = torch.abs(atte_x)

        output = torch.mul(emb_x, atte_x)

        '''--------------------------------------------------------------------------------'''

        output = torch.unsqueeze(output, 1)

        output_1 = self.conv3_1_1(output)
        output_2 = self.conv3_1_2(output)
        output_3 = self.conv3_1_3(output)
        output_4 = self.conv3_1_4(output)
        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)
        output = torch.relu(output)
        output_skip_connect_x = output

        output_1 = self.conv3_2_1(output)
        output_2 = self.conv3_2_2(output)
        output_3 = self.conv3_2_3(output)
        output_4 = self.conv3_2_4(output)
        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)
        output = torch.sigmoid(output)

        output_1 = self.conv3_3_1(output)
        output_2 = self.conv3_3_2(output)
        output_3 = self.conv3_3_3(output)
        output_4 = self.conv3_3_4(output)
        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)
        output = torch.sigmoid(output)
        output = output + output_skip_connect_x

        output_1 = self.conv3_4_1(output)
        output_2 = self.conv3_4_2(output)
        output_3 = self.conv3_4_3(output)
        output_4 = self.conv3_4_4(output)
        output = torch.cat((output_1, output_2, output_3, output_4), dim=1)
        output = self.conv1_squeeze3(output)

        output = torch.squeeze(output, 1)

        return output

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
    def __init__(self,c, k, T):
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
