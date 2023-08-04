import torch
from torch import nn


class ResBlk(nn.Module):
    def __init__(self, kernel_size, in_channels=64, out_channels=64):
        assert kernel_size % 2 == 1  # kernel_size must be odd
        super(ResBlk, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=4, padding=padding, bias=False)

        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=4, padding=padding)
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
    def __init__(self, kernel_size: int, n_res_blks: int):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=kernel_size, padding="same", bias=False)
        self.batchnorm = nn.BatchNorm1d(64, track_running_stats=False)
        self.relu = nn.ReLU()

        self.res_layers = nn.Sequential()
        out_channels = 64
        out_sequence_length = 4096
        for i in range(n_res_blks):
            in_channels = 64 + (i // 2) * 64
            out_channels = 64 + ((i+1) // 2) * 64
            self.res_layers.append(ResBlk(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels))
            out_sequence_length //= 4

        self.linear = torch.nn.Linear(out_channels*out_sequence_length, 1, bias=False)


    def forward(self, x):
        # Preprocessing
        #assert len(x.shape) == 3 and x.shape[1] == 4096 and x.shape[2] == 8
        x = x.transpose(2,1)
        #assert len(x.shape) == 3 and x.shape[1] == 8 and x.shape[2] == 4096

        # Block 1:
        x = self.conv(x)
        #assert len(x.shape) == 3 and x.shape[1] == 64 and x.shape[2] == 4096
        x = self.batchnorm(x)
        x = self.relu(x)

        # Block 2: ResBlks
        x = torch.stack([x, x], dim=0)
        x = self.res_layers(x)
        x = x[1]
        # need to flatten last two dimensions => 32 x 192*4096
        x = x.view(x.size(0), -1)

        # Block 3:
        x = self.linear(x)

        return x