from torch import nn
import models.block as mb


class MS(nn.Module):
    """
    特征提取器
    """
    def __init__(self):
        super(MS, self).__init__()
        self.conv_block = mb.ConvBlock(kernel_size=(3, 3), out_channels=128, padding=1, in_channels=1)
        self.max_pool = mb.MaxPooling(num_feature=128)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.max_pool(x)
        return x
