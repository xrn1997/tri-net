import torch.nn as nn


class ConvBlock(nn.Module):
    """
    卷积块
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=True)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        return x


class ResidualBlock(nn.Module):
    """
    残差块
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=True)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x) + x)
        x = self.leaky_relu(self.conv3(x))
        return x


class MaxPooling(nn.Module):
    """
    最大池化
    """
    def __init__(self, num_feature):
        super(MaxPooling, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.drop_out = nn.Dropout2d(p=0.5)
        self.bn = nn.BatchNorm2d(num_feature)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.drop_out(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class AvgPooling(nn.Module):
    """
    平均池化
    """
    def __init__(self, num_feature):
        super(AvgPooling, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(num_feature)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x
