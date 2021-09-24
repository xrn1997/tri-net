import torch.nn as nn
import models.block as mb


class M1(nn.Module):
    """
    标签预测模型M1
    """

    def __init__(self):
        super(M1, self).__init__()
        self.block1 = mb.ConvBlock(in_channels=128, out_channels=256, kernel_size=(5, 5), padding=2)
        self.max_pool = mb.MaxPooling(num_feature=256)
        self.block2 = mb.ConvBlock(in_channels=256, out_channels=512, kernel_size=(5, 5), padding=1)
        self.block3 = mb.ConvBlock(in_channels=512, out_channels=512, kernel_size=(1, 1), padding=0)
        self.avg_pool = mb.AvgPooling(num_feature=512)
        self.soft_max = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block3(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)

        return x


class M2(nn.Module):
    """
    标签预测模型M2
    """

    def __init__(self):
        super(M2, self).__init__()
        self.block1 = mb.ConvBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.max_pool = mb.MaxPooling(num_feature=256)
        self.block2 = mb.ConvBlock(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0)
        self.block3 = mb.ConvBlock(in_channels=512, out_channels=512, kernel_size=(1, 1), padding=0)
        self.avg_pool = mb.AvgPooling(num_feature=512)
        self.soft_max = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block3(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x


class M3(nn.Module):
    """
    标签预测模型M3
    """

    def __init__(self):
        super(M3, self).__init__()
        self.block1 = mb.ResidualBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.max_pool = mb.MaxPooling(num_feature=256)
        self.block2 = mb.ResidualBlock(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1)
        self.max_pool2 = mb.MaxPooling(num_feature=512)
        self.block3 = mb.ConvBlock(in_channels=512, out_channels=512, kernel_size=(1, 1), padding=0)
        self.avg_pool = mb.AvgPooling(num_feature=512)
        self.soft_max = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(9 * 512, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.block1(x)
        x = self.max_pool(x)

        x = self.block2(x)
        x = self.max_pool2(x)

        x = self.block3(x)
        x = self.block3(x)
        x = self.avg_pool(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.soft_max(x)
        return x
