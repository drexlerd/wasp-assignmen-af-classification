import torch
from torch import nn

class ResBlk(nn.Module):
    def __init__(self, name="", in_channels=64, out_channels=64):
        super(ResBlk, self).__init__()
        self.name = name
        
        track_running_stats = False

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=15, padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=15, stride=4, padding=7, bias=False)
        
        self.max_pool = nn.MaxPool1d(kernel_size=15, stride=4, padding=7)  # n = (k-1)/2 where k is kernel_size: (15-1)/2 = 
        self.conv11 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding="same", bias=False)
        
    def forward(self, x):
        assert len(x.shape) == 4 and x.shape[0] == 2
        x1 = x[0, :, :, :]
        x2 = x[1, :, :, :]
        
        assert len(x1.shape) == 3 and len(x2.shape) == 3
        
        x1 = self.max_pool(x1)
        x1 = self.conv11(x1)
        
        x2 = self.conv1(x2)
        x2 = self.batchnorm1(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.conv2(x2)
   
        assert x1.shape == x2.shape
        
        x = x1 + x2
        
        assert len(x.shape) == 3
        
        x1 = x
        
        x2 = x
        x2 = self.batchnorm1(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        x = torch.stack([x1, x2], dim=0)
        return x
        
        

class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
        
        track_running_stats = False

        self.conv = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=15, padding="same", bias=False)
        self.batchnorm = nn.BatchNorm1d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU()
        
        self.res_lay1 = ResBlk(name="ResBlk1", in_channels=64, out_channels=64)
        self.res_lay2 = ResBlk(name="ResBlk2", in_channels=64, out_channels=128)
        self.res_lay3 = ResBlk(name="ResBlk3", in_channels=128, out_channels=128)
        self.res_lay4 = ResBlk(name="ResBlk4", in_channels=128, out_channels=192)

        self.linear = torch.nn.Linear(192*16, 1, bias=False)
        
        
    def forward(self, x):
        # Preprocessing
        assert len(x.shape) == 3 and x.shape[1] == 4096 and x.shape[2] == 8
        x = x.transpose(2,1)
        assert len(x.shape) == 3 and x.shape[1] == 8 and x.shape[2] == 4096

        # Block 1:
        x = self.conv(x)
        assert len(x.shape) == 3 and x.shape[1] == 64 and x.shape[2] == 4096     
        x = self.batchnorm(x)
        x = self.relu(x)

        # Block 2: ResBlks 
        x = torch.stack([x, x], dim=0)
        assert len(x.shape) == 4 and x.shape[0] == 2 and x.shape[2] == 64 and x.shape[3] == 4096
        x = self.res_lay1(x)
        assert len(x.shape) == 4 and x.shape[0] == 2 and x.shape[2] == 64 and x.shape[3] == 1024
        x = self.res_lay2(x)
        assert len(x.shape) == 4 and x.shape[0] == 2 and x.shape[2] == 128 and x.shape[3] == 256
        x = self.res_lay3(x)
        assert len(x.shape) == 4 and x.shape[0] == 2 and x.shape[2] == 128 and x.shape[3] == 64
        x = self.res_lay4(x)
        assert len(x.shape) == 4 and x.shape[0] == 2 and x.shape[2] == 192 and x.shape[3] == 16
        x = x[1]
        assert len(x.shape) == 3 and x.shape[1] == 192 and x.shape[2] == 16

        # need to flatten last two dimensions => 32 x 192*4096
        x = x.view(x.size(0), -1)
        
        # Block 3:
        x = self.linear(x)

        return x